# Copyright 2024 Anirban Basu

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module contains the webapp for the application."""

try:
    from icecream import ic
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa

from dotenv import load_dotenv
import gradio as gr


class GradioApp:
    """This class represents the Gradio webapp for the application."""

    PROJECT_LOGO_PATH = "assets/logo.svg"

    LABEL_THEME_TOGGLE = "Toggle theme"
    LABEL_SHOW_SIDEBAR = "Show sidebar"
    LABEL_HIDE_SIDEBAR = "Hide sidebar"

    CSS_CLASS_DIV_VERTICAL_ALIGNED = "div-vertical-aligned"
    CSS_CLASS_DIV_RIGHT_ALIGNED = "div-right-aligned"
    CSS_CLASS_BUTTON_FIT_TRANSPARENT = "button-fit-transparent"
    CSS_CLASS_DIV_OUTLINED = "div-outlined"
    CSS_CLASS_DIV_PADDED = "div-padded"

    CSS_GRADIO_APP = f"""
        .{CSS_CLASS_DIV_VERTICAL_ALIGNED} {{
            margin-top: auto;
            margin-bottom: auto;
        }}

        .{CSS_CLASS_DIV_RIGHT_ALIGNED} > * {{
            margin-left: auto;
            margin-right: 0;
        }}

        .{CSS_CLASS_BUTTON_FIT_TRANSPARENT} {{
            width: fit-content;
            background: transparent;
        }}

        .{CSS_CLASS_DIV_OUTLINED} {{
            outline: dotted;
        }}

        .{CSS_CLASS_DIV_PADDED} {{
            padding: 1rem;
        }}
    """

    JS_DARK_MODE_TOGGLE = """
        () => {
            document.body.classList.toggle('dark');
            document.querySelector('gradio-app').style.background = 'var(--body-background-fill)';
        }
    """

    def __init__(self):
        ic(load_dotenv())
        self.interface = self.create_interface()

    def create_interface(self):
        """Construct the Gradio user interface and make it available through the `interface` property of this class."""
        with gr.Blocks(
            # See theming guide at https://www.gradio.app/guides/theming-guide
            fill_width=True,
            fill_height=True,
            css=GradioApp.CSS_GRADIO_APP,
            # Setting the GRADIO_ANALYTICS_ENABLED environment variable to "True" will have no effect.
            analytics_enabled=False,
            # Delete the cache content every day that is older than a day
            delete_cache=(86400, 86400),
        ) as self.interface:
            gr.set_static_paths(paths=[GradioApp.PROJECT_LOGO_PATH])
            self._sidebar_state = False

            with gr.Row(equal_height=True):
                with gr.Column(
                    scale=10, elem_classes=[GradioApp.CSS_CLASS_DIV_VERTICAL_ALIGNED]
                ):
                    gr.HTML(
                        """
                            <img
                                width="384"
                                height="192"
                                style="filter: invert(0.5);"
                                alt="dqa logo"
                                src="https://raw.githubusercontent.com/anirbanbasu/dqa/master/assets/logo.svg" />
                        """,
                        # /file={GradioApp.PROJECT_LOGO_PATH}
                    )
                with gr.Column(
                    scale=2,
                    elem_classes=[
                        GradioApp.CSS_CLASS_DIV_VERTICAL_ALIGNED,
                        GradioApp.CSS_CLASS_DIV_RIGHT_ALIGNED,
                    ],
                ):
                    btn_theme_toggle = gr.Button(
                        GradioApp.LABEL_THEME_TOGGLE,
                        size="sm",
                    )
                    btn_sidebar_toggle = gr.Button(
                        GradioApp.LABEL_SHOW_SIDEBAR,
                        size="sm",
                    )

            with gr.Row(equal_height=True):
                with gr.Column(visible=self._sidebar_state, scale=1) as sidebar:
                    gr.HTML(
                        """
                            <h1>Sidebar</h1>
                            <p>This is a sidebar to contain configurable settings. It can be toggled on and off.</p>
                        """
                    )
                with gr.Column(scale=2):
                    gr.HTML(
                        """
                            <h1>Main content</h1>
                            <p>This is the main content area.</p>
                        """
                    )

            # Component actions
            btn_theme_toggle.click(
                fn=None,
                js=GradioApp.JS_DARK_MODE_TOGGLE,
                api_name=False,
            )

            @btn_sidebar_toggle.click(
                api_name=False, outputs=[sidebar, btn_sidebar_toggle]
            )
            def toggle_sidebar_state():
                self._sidebar_state = not self._sidebar_state
                return (
                    gr.update(visible=self._sidebar_state),
                    gr.update(
                        value=(
                            GradioApp.LABEL_HIDE_SIDEBAR
                            if self._sidebar_state
                            else GradioApp.LABEL_SHOW_SIDEBAR
                        )
                    ),
                )

    def run(self):
        """Run the Gradio app by launching a server."""
        self.create_interface()
        allowed_static_file_paths = [
            GradioApp.PROJECT_LOGO_PATH,
        ]
        ic(allowed_static_file_paths)
        self.interface.queue().launch(
            show_api=True,
            show_error=True,
            allowed_paths=allowed_static_file_paths,
            # Enable monitoring only for debugging purposes?
            # enable_monitoring=True,
        )


if __name__ == "__main__":
    app = GradioApp()
    app.run()
