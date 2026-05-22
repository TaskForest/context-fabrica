"""TypeScript/TSX AST knowledge extractor.

Uses tree-sitter with the TypeScript grammar to extract classes, functions,
imports, interfaces, type aliases, inheritance, and call relationships from
TypeScript and TSX source files.
"""
from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from pathlib import Path
import re
from typing import Any

from ..models import ExtractionResult, Relation

_TS_SUFFIXES = (".ts", ".tsx")
_FUNCTION_LIKE_NODES = {
    "function_declaration",
    "method_definition",
    "method_signature",
    "arrow_function",
    "function_expression",
}
_BOUNDARY_NODES = {"function_declaration", "method_definition", "method_signature", "class_declaration", "abstract_class_declaration", "interface_declaration"}


class TypeScriptASTExtractor:
    """Extract knowledge from TypeScript and TSX source via tree-sitter."""

    def __init__(self, *, domain: str = "code", confidence: float = 0.9) -> None:
        self._domain = domain
        self._confidence = confidence

    def extract(self, path: Path) -> list[ExtractionResult]:
        """Extract knowledge from all ``.ts`` and ``.tsx`` files under *path*."""
        path = Path(path)
        if path.is_file():
            files = [path] if path.suffix in _TS_SUFFIXES else []
        else:
            files = sorted([*path.rglob("*.ts"), *path.rglob("*.tsx")])

        results: list[ExtractionResult] = []
        for ts_file in files:
            result = self._extract_file(ts_file)
            if result is not None:
                results.append(result)
        return results

    def _extract_file(self, file_path: Path) -> ExtractionResult | None:
        try:
            source = file_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            return None

        parser = _parser_for_suffix(file_path.suffix)
        source_bytes = source.encode("utf-8")
        tree = parser.parse(source_bytes)
        root = tree.root_node
        if getattr(root, "has_error", False):
            return None

        visitor = _TypeScriptVisitor(file_path=file_path, source_bytes=source_bytes)
        visitor.visit(root)

        if not visitor.entities and not visitor.summaries:
            return None

        language = "tsx" if file_path.suffix == ".tsx" else "typescript"
        return ExtractionResult(
            text="\n".join(visitor.summaries),
            source=str(file_path),
            entities=visitor.entities,
            relations=visitor.relations,
            confidence=self._confidence,
            domain=self._domain,
            tags=["ast-extracted", language],
            metadata={
                "language": language,
                "source_file": str(file_path),
                "classes": visitor.class_names,
                "functions": visitor.function_names,
                "imports": visitor.import_names,
                "interfaces": visitor.interface_names,
                "types": visitor.type_names,
            },
        )


class _TypeScriptVisitor:
    def __init__(self, *, file_path: Path, source_bytes: bytes) -> None:
        self.file_path = str(file_path)
        self._module_name = file_path.stem
        self._source_bytes = source_bytes
        self.entities: list[str] = []
        self.relations: list[Relation] = []
        self.summaries: list[str] = []
        self.class_names: list[str] = []
        self.function_names: list[str] = []
        self.import_names: list[str] = []
        self.interface_names: list[str] = []
        self.type_names: list[str] = []
        self._current_class: str | None = None
        self._current_interface: str | None = None

    def visit(self, node: Any) -> None:
        handlers = {
            "program": self.visit_program,
            "class_declaration": self.visit_class_declaration,
            "abstract_class_declaration": self.visit_class_declaration,
            "interface_declaration": self.visit_interface_declaration,
            "type_alias_declaration": self.visit_type_alias_declaration,
            "function_declaration": self.visit_function_declaration,
            "method_definition": self.visit_method_definition,
            "method_signature": self.visit_method_signature,
            "lexical_declaration": self.visit_lexical_declaration,
            "import_statement": self.visit_import_statement,
        }
        handler = handlers.get(node.type)
        if handler is not None:
            handler(node)
            return
        for child in node.named_children:
            self.visit(child)

    def visit_program(self, node: Any) -> None:
        module_comment = _leading_module_comment(self._source_bytes)
        if module_comment:
            self.summaries.append(f"Module {self._module_name}: {module_comment}")
        for child in node.named_children:
            self.visit(child)

    def visit_class_declaration(self, node: Any) -> None:
        name = _name_of(node.child_by_field_name("name"), self._source_bytes)
        if not name:
            return

        self.entities.append(name)
        self.class_names.append(name)

        parts = [f"Class {name}"]
        base_names = self._class_bases(node)
        if base_names:
            parts.append(f"inherits {', '.join(base_names)}")
            for base_name in base_names:
                self.entities.append(base_name)
                self.relations.append(Relation(name, "inherits", base_name))

        doc = _leading_comment(self._source_bytes, node.start_byte)
        if doc:
            parts.append(f"— {doc}")

        methods = self._class_method_names(node)
        if methods:
            parts.append(f"methods: {', '.join(methods)}")

        self.summaries.append(". ".join(parts) + ".")

        previous = self._current_class
        self._current_class = name
        body = node.child_by_field_name("body")
        if body is not None:
            for child in body.named_children:
                self.visit(child)
        self._current_class = previous

    def visit_interface_declaration(self, node: Any) -> None:
        name = _name_of(node.child_by_field_name("name"), self._source_bytes)
        if not name:
            return

        self.entities.append(name)
        self.interface_names.append(name)

        parts = [f"Interface {name}"]
        doc = _leading_comment(self._source_bytes, node.start_byte)
        if doc:
            parts.append(f"— {doc}")

        methods = self._interface_method_names(node)
        if methods:
            parts.append(f"methods: {', '.join(methods)}")

        self.summaries.append(". ".join(parts) + ".")

        previous = self._current_interface
        self._current_interface = name
        body = node.child_by_field_name("body")
        if body is not None:
            for child in body.named_children:
                self.visit(child)
        self._current_interface = previous

    def visit_type_alias_declaration(self, node: Any) -> None:
        name = _name_of(node.child_by_field_name("name"), self._source_bytes)
        value = _type_alias_value(node, self._source_bytes)
        if not name:
            return

        self.entities.append(name)
        self.type_names.append(name)
        summary = f"Type alias {name}"
        if value:
            summary += f" = {value}"
        self.summaries.append(summary + ".")

    def visit_function_declaration(self, node: Any) -> None:
        name = _name_of(node.child_by_field_name("name"), self._source_bytes)
        if not name:
            return
        self._record_function(name=name, node=node, kind="Function")

    def visit_method_definition(self, node: Any) -> None:
        method_name = _name_of(node.child_by_field_name("name"), self._source_bytes)
        owner = self._current_class
        if not method_name or not owner:
            return
        qualified = f"{owner}.{method_name}"
        self.relations.append(Relation(owner, "has_method", qualified))
        self._record_function(name=qualified, node=node, kind="Method")

    def visit_method_signature(self, node: Any) -> None:
        method_name = _name_of(node.child_by_field_name("name"), self._source_bytes)
        owner = self._current_interface
        if not method_name or not owner:
            return
        qualified = f"{owner}.{method_name}"
        self.relations.append(Relation(owner, "has_method", qualified))
        self._record_function(name=qualified, node=node, kind="Method signature")

    def visit_lexical_declaration(self, node: Any) -> None:
        if self._current_class or self._current_interface:
            return
        for child in node.named_children:
            if child.type != "variable_declarator":
                continue
            name_node = child.child_by_field_name("name")
            value_node = child.child_by_field_name("value")
            if value_node is None or value_node.type not in {"arrow_function", "function_expression"}:
                continue
            name = _name_of(name_node, self._source_bytes)
            if name:
                self._record_function(name=name, node=value_node, kind="Function")

    def visit_import_statement(self, node: Any) -> None:
        source_node = node.child_by_field_name("source")
        module_name = _string_content(source_node, self._source_bytes)
        if not module_name:
            return

        imported_names = self._imported_names(node)
        if not imported_names:
            imported_names = [module_name]

        for imported_name in imported_names:
            self.import_names.append(imported_name)
            self.entities.append(imported_name)
            self.relations.append(Relation(self._module_name, "imports", imported_name))

    def _record_function(self, *, name: str, node: Any, kind: str) -> None:
        self.entities.append(name)
        self.function_names.append(name)

        parts = [f"{kind} {name}"]
        doc = _leading_comment(self._source_bytes, node.start_byte)
        if doc:
            parts.append(f"— {doc}")

        params_node = node.child_by_field_name("parameters")
        params = _parameter_names(params_node, self._source_bytes)
        if params:
            parts.append(f"params: {', '.join(params)}")

        self.summaries.append(". ".join(parts) + ".")

        body = node.child_by_field_name("body")
        self._collect_call_relations(name, body or node)

        if body is not None:
            for child in body.named_children:
                if child.type in {"function_declaration", "lexical_declaration", "class_declaration", "abstract_class_declaration", "interface_declaration"}:
                    self.visit(child)

    def _collect_call_relations(self, owner: str, node: Any) -> None:
        for child in node.named_children:
            if child.type == "call_expression":
                callee = child.child_by_field_name("function") or (child.named_children[0] if child.named_children else None)
                call_name = _name_of(callee, self._source_bytes)
                if call_name and call_name != owner:
                    self.relations.append(Relation(owner, "calls", call_name))
            if child.type in _BOUNDARY_NODES:
                continue
            self._collect_call_relations(owner, child)

    def _class_bases(self, node: Any) -> list[str]:
        base_names: list[str] = []
        for child in node.named_children:
            if child.type != "class_heritage":
                continue
            for heritage_child in child.named_children:
                candidate = _name_of(heritage_child, self._source_bytes)
                if candidate and candidate != "extends":
                    base_names.append(candidate)
        return base_names

    def _class_method_names(self, node: Any) -> list[str]:
        body = node.child_by_field_name("body")
        if body is None:
            return []
        methods: list[str] = []
        for child in body.named_children:
            if child.type != "method_definition":
                continue
            method_name = _name_of(child.child_by_field_name("name"), self._source_bytes)
            if method_name:
                methods.append(method_name)
        return methods

    def _interface_method_names(self, node: Any) -> list[str]:
        body = node.child_by_field_name("body")
        if body is None:
            return []
        methods: list[str] = []
        for child in body.named_children:
            if child.type != "method_signature":
                continue
            method_name = _name_of(child.child_by_field_name("name"), self._source_bytes)
            if method_name:
                methods.append(method_name)
        return methods

    def _imported_names(self, node: Any) -> list[str]:
        names: list[str] = []
        for child in node.named_children:
            if child.type != "import_clause":
                continue
            for grandchild in child.named_children:
                if grandchild.type in {"identifier", "namespace_import"}:
                    name = _name_of(grandchild, self._source_bytes)
                    if name:
                        names.append(name)
                elif grandchild.type == "named_imports":
                    for specifier in grandchild.named_children:
                        alias = specifier.child_by_field_name("alias")
                        name_node = alias or specifier.child_by_field_name("name")
                        if name_node is None and specifier.named_children:
                            name_node = specifier.named_children[-1]
                        name = _name_of(name_node, self._source_bytes)
                        if name:
                            names.append(name)
        return names


@lru_cache(maxsize=2)
def _parser_for_suffix(suffix: str) -> Any:
    try:
        tree_sitter = import_module("tree_sitter")
        tree_sitter_typescript = import_module("tree_sitter_typescript")
    except ImportError as exc:  # pragma: no cover - exercised via runtime install guidance
        raise RuntimeError(
            "TypeScript parsing requires optional dependencies. Install with "
            "`pip install 'context-fabrica[typescript]'` or "
            "`pip install tree-sitter tree-sitter-typescript`."
        ) from exc

    language_fn = tree_sitter_typescript.language_tsx if suffix == ".tsx" else tree_sitter_typescript.language_typescript
    return tree_sitter.Parser(tree_sitter.Language(language_fn()))


def _parameter_names(node: Any | None, source_bytes: bytes) -> list[str]:
    if node is None:
        return []
    names: list[str] = []
    for child in node.named_children:
        name = _parameter_name(child, source_bytes)
        if name and name != "this":
            names.append(name)
    return names


def _parameter_name(node: Any, source_bytes: bytes) -> str | None:
    if node.type in {"identifier", "property_identifier", "type_identifier", "this"}:
        return _node_text(node, source_bytes)

    pattern = node.child_by_field_name("pattern")
    if pattern is not None:
        candidate = _parameter_name(pattern, source_bytes)
        if candidate:
            return candidate

    for child in node.named_children:
        if child.type in {"type_annotation", "predefined_type", "generic_type"}:
            continue
        candidate = _parameter_name(child, source_bytes)
        if candidate:
            return candidate
    return None


def _type_alias_value(node: Any, source_bytes: bytes) -> str | None:
    name_node = node.child_by_field_name("name")
    for child in node.named_children:
        if name_node is not None and child.start_byte == name_node.start_byte and child.end_byte == name_node.end_byte:
            continue
        return _node_text(child, source_bytes)
    return None


def _name_of(node: Any | None, source_bytes: bytes) -> str | None:
    if node is None:
        return None
    if node.type in {"identifier", "property_identifier", "type_identifier", "private_property_identifier", "this", "super"}:
        return _node_text(node, source_bytes)
    if node.type == "member_expression":
        object_node = node.child_by_field_name("object") or (node.named_children[0] if node.named_children else None)
        property_node = node.child_by_field_name("property") or (node.named_children[-1] if node.named_children else None)
        object_name = _name_of(object_node, source_bytes)
        property_name = _name_of(property_node, source_bytes)
        if object_name and property_name:
            return f"{object_name}.{property_name}"
        return object_name or property_name
    if node.type == "subscript_expression":
        return _name_of(node.child_by_field_name("object") or (node.named_children[0] if node.named_children else None), source_bytes)
    if node.type == "namespace_import":
        for child in node.named_children:
            candidate = _name_of(child, source_bytes)
            if candidate:
                return candidate
    for child in node.named_children:
        candidate = _name_of(child, source_bytes)
        if candidate:
            return candidate
    return None


def _string_content(node: Any | None, source_bytes: bytes) -> str | None:
    if node is None:
        return None
    text = _node_text(node, source_bytes)
    return text.strip('"\'`')


def _node_text(node: Any, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")


def _leading_module_comment(source_bytes: bytes) -> str | None:
    head = source_bytes[:800].decode("utf-8", errors="ignore")
    match = re.match(r"\s*/\*\*(.*?)\*/", head, re.DOTALL)
    if match:
        return _clean_comment(match.group(1))
    return None


def _leading_comment(source_bytes: bytes, start_byte: int) -> str | None:
    prefix = source_bytes[max(0, start_byte - 1000):start_byte].decode("utf-8", errors="ignore")
    block_match = re.search(r"/\*\*(.*?)\*/\s*$", prefix, re.DOTALL)
    if block_match:
        return _clean_comment(block_match.group(1))

    line_match = re.search(r"(?://[^\n]*\n\s*)+$", prefix)
    if not line_match:
        return None
    lines = [line.strip()[2:].strip() for line in line_match.group(0).splitlines() if line.strip().startswith("//")]
    if not lines:
        return None
    return " ".join(lines)


def _clean_comment(raw: str) -> str:
    lines = []
    for line in raw.splitlines():
        cleaned = line.strip()
        if cleaned.startswith("*"):
            cleaned = cleaned[1:].strip()
        if cleaned:
            lines.append(cleaned)
    return lines[0] if lines else ""
