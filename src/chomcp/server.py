from mcp.server.fastmcp import FastMCP
from chonkie import TokenChunker

mcp = FastMCP("Chonkie")

# Everyone starts somewhere and Chonkie is no different. We 
# have humble beginnings, where we start with a simple tool
# that chunks text into smaller chunks using the tokenizer. 
# This is a good starting point for Chonkie and will be 
# improved upon in the future.
@mcp.tool()
def chunk_text(text: str) -> str:
    """Chunk text into smaller chunks."""
    chunker = TokenChunker()
    return chunker(text)

# This would be run when the user runs `chonkie-mcp` from the command line.
# Its the main entry point for the package.
def main():
    """Main entry point for the package."""
    mcp.run()