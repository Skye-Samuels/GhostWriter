from textual.app import App, ComposeResult, RenderResult
from textual.binding import Binding
from .components import InputText, OutputText, Sidebar, Header, Footer, VerticalScrollOutput
from textual.containers import VerticalScroll, Horizontal

class GhostWriterApp(App):
    CSS_PATH = "styles.tcss"
    TITLE = "GhostWriter"
    BINDINGS = [
        Binding(key="q", action="quit", description="Quit the app"),
        Binding(key="d", action="detect", description="Detect for AI content"),
    ]
    ENABLE_COMMAND_PALETTE = False

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            yield InputText()
            yield VerticalScrollOutput()
        yield Sidebar()
        yield Footer()