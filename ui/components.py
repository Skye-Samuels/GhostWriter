from textual.widgets import TextArea, Label, Static, Button, Header, Footer
from textual.containers import VerticalScroll
from textual.app import ComposeResult
from binoculars.detector import Binoculars

# Initialize Binoculars AI detector
binoculars = Binoculars()

class Header(Header):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title = "GhostWriter"

class Footer(Footer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class InputText(TextArea):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.border_title = "Input Text"

class OutputText(Label):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class VerticalScrollOutput(VerticalScroll):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.border_title = "Output Text"

    def compose(self) -> ComposeResult:
        yield OutputText()

class Sidebar(Static):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.border_title = "Results"

    def compose(self) -> ComposeResult:
        yield DetectorResults()
        yield DetectButton()

class DetectorResults(Static):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.renderable = "Verdict: ???\nScore: ???"

class DetectButton(Button):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label = "Detect"
        self.shortcut = "d"

    def on_click(self):
        input_text_widget = self.app.query_one(InputText)
        input_text = input_text_widget.text

        # Use Binoculars to analyze the text
        verdict = binoculars.predict(input_text)
        binoculars_score = binoculars.compute_score(input_text)

        output_text_widget = self.app.query_one(OutputText)
        detector_results_widget = self.app.query_one(DetectorResults)

        if verdict == "Human":
            output_text_widget.update(f"[green]{input_text}[/]")
            detector_results_widget.update(f"Verdict: [green]{verdict}[/]\nScore: [green]{binoculars_score:.2f}[/]")
            detector_results_widget.parent.styles.border = ("round", "green")
        else:
            output_text_widget.update(f"[red]{input_text}[/]")
            detector_results_widget.update(f"Verdict: [red]{verdict}[/]\nScore: [red]{binoculars_score:.2f}[/]")
            detector_results_widget.parent.styles.border = ("round", "red")
        
        output_text_widget.parent.refresh()
