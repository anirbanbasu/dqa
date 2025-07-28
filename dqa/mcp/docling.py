# Note: there is a Docling MCP server, however, we wrap around limited functionality here.
from typing import Annotated
from fastmcp import FastMCP
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from dqa.common import ic


# from docling.utils.model_downloader import download_models
# download_models()

app = FastMCP(
    instructions="A collection of document processing tools.",
    on_duplicate_prompts="error",
    on_duplicate_resources="error",
    on_duplicate_tools="error",
)


@app.tool(
    tags=["document", "processing", "convert"],
)
def read_arxiv_preprint(
    source_url: Annotated[str, "URL of the arXiv pre-print paper"],
) -> str:
    """
    Reads an arXiv pre-print paper and converts it to text formatted as Markdown.
    """
    pipeline_options = PdfPipelineOptions(enable_remote_services=True)
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    try:
        result = converter.convert(source_url)
        return result.document.export_to_markdown()
    except Exception as e:
        ic(e)
        return ""
