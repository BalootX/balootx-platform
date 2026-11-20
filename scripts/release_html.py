"""Extract script text for the release verifiers' JavaScript syntax checks."""

from html.parser import HTMLParser


class _ScriptParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.blocks: list[str] = []
        self._body: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "script":
            self._body = []

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "script":
            raise ValueError("script element requires a closing tag")

    def handle_data(self, data: str) -> None:
        if self._body is not None:
            self._body.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._body is not None:
            self.blocks.append("".join(self._body))
            self._body = None


def script_blocks(html: str) -> list[str]:
    parser = _ScriptParser()
    parser.feed(html)
    parser.close()
    if parser._body is not None:
        raise ValueError("unterminated script element")
    return parser.blocks
