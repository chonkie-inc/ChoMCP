from mcp.server.fastmcp import FastMCP
from chonkie import (  #type: ignore
    TokenChunker, 
    SentenceChunker, 
    RecursiveChunker,
    CodeChunker,
    SemanticChunker,
    SDPMChunker,
    NeuralChunker,
    SlumberChunker, 
    LateChunker,
)
from typing import List, Optional, Literal, Union, Dict, Any
import logging
import sys
import traceback

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

mcp = FastMCP(name="Chonkie", instructions="A server for Chonkie text chunking tools.", port=8003)

# Everyone starts somewhere and Chonkie is no different. We 
# have humble beginnings, where we start with a simple tool
# that chunks text into smaller chunks using the tokenizer. 
# This is a good starting point for Chonkie and will be 
# improved upon in the future.
@mcp.tool()
async def token_chunker(text: str,
                  tokenizer: str = "gpt2",
                  chunk_size: int = 512,
                  chunk_overlap: int = 0) -> List[str]:
    """Split the text into smaller chunks based on the token counts. 

    Args:
        text (str): The text to chunk.
        tokenizer (str): The tokenizer to use for this. Loads the "gpt2" tokenizer by default.
        chunk_size (int): The size of the chunks to split the text into.
        chunk_overlap (int): The overlap between the chunks.

    Returns:
        List[str]: The text split into smaller chunks.
    """
    chunker = TokenChunker(tokenizer=tokenizer,
                           chunk_size=chunk_size, 
                           chunk_overlap=chunk_overlap, 
                           return_type="texts")
    chunks =  chunker(text)
    return chunks

# Okay, that's a good start. Now we can include all the other chunking methods
# that Chonkie supports.
@mcp.tool()
async def sentence_chunker(text: str,
                     tokenizer: str = "gpt2",
                     chunk_size: int = 512,
                     min_sentences_per_chunk: int = 1,
                     min_characters_per_sentence: int = 12,
                     delim: List[str] = [". ", "? ", "! ", "\n"],
                     include_delim: Optional[Literal["prev", "next"]] = "prev") -> List[str]:
    """Chunk the text based on the sentence boundaries. 

    SentenceChunker is smarter than the TokenChunker and a good way to ensure that the chunks are
    not split mid-word because TokenChunker can split on subwords. It usually does a better job 
    on downstream recall when the chunk is searched for and is fairly fast.
    
    Args:
        text (str): The text to chunk.
        tokenizer (str): The tokenizer to use for this. Loads the "gpt2" tokenizer by default.
        chunk_size (int): The size of the chunks to split the text into. Defaults to 512.   
        min_sentences_per_chunk (int): The minimum number of sentences per chunk. Defaults to 1.
        min_characters_per_sentence (int): The minimum number of characters per sentence. Defaults to 12.
        delim (List[str]): The delimiters to use for the sentence boundaries. Defaults to [". ", "? ", "! ", "\n"].
        include_delim (Optional[Literal["prev", "next"]]): Whether to include the delimiters in the chunks. Defaults to "prev".

    Returns:
        List[str]: The text split into smaller chunks.
    """
    chunker = SentenceChunker(tokenizer_or_token_counter=tokenizer, 
                              chunk_size=chunk_size,
                              min_sentences_per_chunk=min_sentences_per_chunk,
                              min_characters_per_sentence=min_characters_per_sentence,
                              delim=delim, 
                              include_delim=include_delim,
                              return_type="texts")
    chunks = chunker(text)
    return chunks

# We can also include the recursive chunker. This is a good way to chunk the text
# into smaller chunks.
@mcp.tool()
async def recursive_chunker(text: str,
                      recipe: str = "default", 
                      lang: str = "en",
                      tokenizer: str = "gpt2",
                      chunk_size: int = 512, 
                      min_characters_per_chunk: int = 24) -> List[str]:
    """Chunk the text based on the structural cues, using a recipe. 

    RecursiveChunker is the best default way to start for a random piece of text. It uses
    structural cues mentioned in the recipe to chunk the text into smaller chunks. The chunks come
    out pretty good and is slightly faster than SentenceChunker.

    When the recipe is set to default, it splits the text based on 5 levels of structural cues. The paragraphs (based on double line breaks), the sentences (based on the sentence boundaries), the sub-sentences (based on the punctuation), the words (based on the word boundaries), and the characters (based on the character boundaries).

    When the recipe is set to markdown, it splits the text based on the markdown headers first and then uses the default recipe for the rest of the text.

    It only goes to a lower level if the text has at a higher level has not been split up enough and is larger than the chunk size as asked by the user.

    Args:
        text (str): The text to chunk.
        recipe (str): The recipe to use for the chunking. Loads the "default" recipe by default. Currently, only 'default' and 'markdown' are supported.
        lang (str): The language of the text. Defaults to "en". Currently, supports ['en', 'zh', 'jp', 'ko', 'hi]
        tokenizer (str): The tokenizer to use for this. Loads the "gpt2" tokenizer by default.
        chunk_size (int): The size of the chunks to split the text into. Defaults to 512.
        min_characters_per_chunk (int): The minimum number of characters per chunk. Defaults to 24.

    Returns:
        List[str]: The text split into smaller chunks.
    """
    chunker = RecursiveChunker.from_recipe(name=recipe, lang=lang, tokenizer_or_token_counter=tokenizer, chunk_size=chunk_size, min_characters_per_chunk=min_characters_per_chunk, return_type="texts")
    chunks = chunker(text)
    return chunks

# We can also include the code chunker. This is a good way to chunk the text
# into smaller chunks.
@mcp.tool()
async def code_chunker(text: str,
                 tokenizer: str = "gpt2",
                 chunk_size: int = 512, 
                 language: str = "auto") -> List[str]:   
    """Chunk the text based on a Abstract Syntax Tree (AST) made based on the code language grammer. 

    CodeChunker is the best way to chunk code files because it uses a AST to chunk the text based on the code language grammer. This results in chunks that are structurally meaningful, trying to keep code portions in tact.

    Args:
        text (str): The text to chunk.
        tokenizer (str): The tokenizer to use for this. Loads the "gpt2" tokenizer by default.
        chunk_size (int): The size of the chunks to split the text into. Defaults to 512.
        language (str): The language of the text. Defaults to "auto". Currently, supports ['auto', 'python', 'javascript', 'java', 'c', 'cpp', 'go', 'rust', 'php', 'ruby', 'swift', 'kotlin', 'scala', 'haskell', 'ocaml', 'lua', 'erlang', 'elixir', 'dart', 'typescript', 'r', 'matlab', 'julia', 'rust', 'php', 'ruby', 'swift', 'kotlin', 'scala', 'haskell', 'ocaml', 'lua', 'erlang', 'elixir', 'dart', 'typescript', 'r', 'matlab', 'julia']. In `auto` mode, the language is detected automatically based on the content of the text. Recommended to pass the language for better performance since `auto` mode takes some time to detect the language.

    Returns:
        List[str]: The text split into smaller chunks.
    """
    chunker = CodeChunker(tokenizer_or_token_counter=tokenizer, 
                          chunk_size=chunk_size, 
                          language=language,
                          return_type="texts")
    chunks = chunker(text)
    return chunks

# We can also include the semantic chunker. This is a good way to chunk the text
# into smaller chunks.
@mcp.tool()
async def semantic_chunker(text: str,
                     embedding_model: str = "minishlab/potion-base-8M",
                     mode: str = "window",
                     threshold: Union[str, float, int] = "auto",
                     chunk_size: int = 512, 
                     similarity_window: int = 1, 
                     min_sentences: int = 1, 
                     min_chunk_size: int = 2, 
                     min_characters_per_sentence: int = 12, 
                     threshold_step: float = 0.01, 
                     delim: List[str] = [". ", "? ", "! ", "\n"],
                     include_delim: Optional[Literal["prev", "next"]] = "prev",
                     tokenizer: str = "gpt2") -> List[str]:   
    """Chunk the text based on the semantic similarity of the sentences. 

    SemanticChunker uses true semantics to chunk the text into smaller chunks. Using a embedding model, it gets the vector embeddings of each of the sentences and compares the cosine similarity of the current sentence with the past few sentences to decide if it group them together or not. If the similarity distance is greater than the threshold, it will group them together.

    The threshold is set between 0 and 1. In `auto` mode, the threshold is calculated based on the chunk_size and min_chunk_size, ensuring the threshold is such the median chunk_size is well within the values. Everytime it updates the threshold in any direction, it will step by the `threshold_step` value. If the threshold is set to a number larger than 1, then it would take that as a percentile value — wherein 90 percentile is a good starting point.
    
    We currently use the `minishlab/potion-base-8M` model for the embedding model. It's a static embedding model which runs super fast on CPU and is reasonably good.

    The mode can be set to `window` or `cumulative`. In `window` mode, the chunker will look at the past few sentences to decide if it should group them together or not. In `cumulative` mode, the chunker will look at all the sentences grouped till the current sentence to decide if it should group them together or not.

    Args:
        text (str): The text to chunk.
        embedding_model (str): The embedding model to use for this. Loads the "minishlab/potion-base-8M" model by default. 
        mode (str): The mode to use for the chunking. Defaults to "window".
        threshold (Union[str, float, int]): The threshold to use for the chunking. Defaults to "auto".
        chunk_size (int): The size of the chunks to split the text into. Defaults to 512.
        similarity_window (int): The window size to use for the chunking. Defaults to 1.
        min_sentences (int): The minimum number of sentences per chunk. Defaults to 1.
        min_chunk_size (int): The minimum number of characters per chunk. Defaults to 2.
        min_characters_per_sentence (int): The minimum number of characters per sentence. Defaults to 12.
        threshold_step (float): The step size to use for the threshold. Defaults to 0.01.
        delim (List[str]): The delimiters to use for the chunking. Defaults to [". ", "? ", "! ", "\n"].
        include_delim (Optional[Literal["prev", "next"]]): Whether to include the delimiters in the chunks. Defaults to "prev".
        tokenizer (str): The tokenizer to use for this. Loads the "gpt2" tokenizer by default.

    Returns:
        List[str]: The text split into smaller chunks.
    """
    chunker = SemanticChunker(embedding_model=embedding_model, 
                              chunk_size=chunk_size, 
                              mode=mode,
                              threshold=threshold,
                              similarity_window=similarity_window,
                              min_sentences=min_sentences,
                              min_chunk_size=min_chunk_size,
                              min_characters_per_sentence=min_characters_per_sentence,
                              threshold_step=threshold_step,
                              delim=delim,
                              include_delim=include_delim,
                              return_type="texts")
    chunks = chunker(text)
    return chunks

# We can also include the SDPM chunker. This is a good way to chunk the text        
# into smaller chunks.
@mcp.tool()
async def sdpm_chunker(text: str,
                tokenizer: str = "gpt2",
                chunk_size: int = 512,
                skip_window: int = 2) -> List[str]:   
    """Chunk the text into smaller chunks using the SDPM (Skip-based Dynamic Pattern Matching) method.

    SDPMChunker is an extension of SemanticChunker that uses a skip-based approach to improve chunking quality.
    It evaluates chunks within a skip window from the current chunk and merges them if they have high semantic
    similarity. This approach helps in better handling long-range dependencies and maintaining context across
    larger spans of text.

    The skip window parameter allows the chunker to look beyond immediate neighbors, which can be particularly
    useful for texts with complex structures or when semantic relationships span across multiple chunks.

    Args:
        text (str): The text to chunk.
        tokenizer (str): The tokenizer to use for this. Loads the "gpt2" tokenizer by default.
        chunk_size (int): The size of the chunks to split the text into. Defaults to 512.

        skip_window (int): The number of chunks to look ahead/behind when evaluating semantic similarity.
                          Defaults to 2. A larger window allows for better handling of long-range dependencies
                          but may increase processing time.

    Returns:
        List[str]: The text split into smaller chunks.
    """
    chunker = SDPMChunker(tokenizer_or_token_counter=tokenizer, 
                          chunk_size=chunk_size,
                          skip_window=skip_window,
                          return_type="texts")
    chunks = chunker(text)
    return chunks

# We can use the Late Chunker to chunk the text into smaller chunks.
@mcp.tool()
async def late_chunker(text: str,
                      embedding_model: str = "sentence-transformers/all-minilm-l6-v2",
                      chunk_size: int = 512) -> List[str]:
    """Chunk the text using the LateChunking method. 

    LateChunking is a method that gets the embeddings of the entire text (or as much as possible) and then uses a recursive method to chunk the text into smaller chunks. This way the embedding due to the action of the attention mechanism gets a full context. 

    Args:
        text (str): The text to chunk.

    Returns:
        List[str]: The text split into smaller chunks.
    """
    chunker = LateChunker(embedding_model=embedding_model, chunk_size=chunk_size)
    chunks = chunker(text)
    chunks = [chunk.to_dict() for chunk in chunks]
    return chunks

# We can also include the neural chunker. This is a good way to chunk the text
# into smaller chunks.
@mcp.tool()
async def neural_chunker(text: str) -> List[str]:   
    """Chunk the text using a Transformer-based Neural Network fine-tuned for chunking.

    NeuralChunker uses a Transformer-based Neural Network fine-tuned for chunking. It uses a 
    BERT-like model with a classification head for each token to predict if the token is the end
    point of a chunk or not. 

    Args:
        text (str): The text to chunk.

    Returns:
        List[str]: The text split into smaller chunks.

    """
    chunker = NeuralChunker(return_type="texts")
    chunks = chunker(text)
    return chunks

# We can also include the slumber chunker. This is a good way to chunk the text
# into smaller chunks.
@mcp.tool()
async  def slumber_chunker(text: str) -> List[str]:   
    """Chunk the text using an LLM that uses the Slumber method to get the ideal split points.

    SlumberChunker uses an LLM that uses the Slumber method to get the ideal split points. It leverages
    the intelligence of the LLM to get the best split points for the text.

    This requires the `GEMINI_API_KEY` environment variable to be set for use. If it's not set, 
    this will throw an error.

    Args:
        text (str): The text to chunk.

    Returns:
        List[str]: The text split into smaller chunks.
    """
    chunker = SlumberChunker(return_type="texts")
    chunks = chunker(text)
    return chunks


# This would be run when the user runs `chonkie-mcp` from the command line.
# Its the main entry point for the package.
# def main():
#     """Main entry point for the package."""
#     logger.info("Starting Chonkie MCP server...")
#     mcp.run(transport="stdio")

# @app.command()
# def run():
#     """Run the Chonkie MCP server."""
#     typer.run(main)

if __name__ == "__main__":
    try:
        print("Starting Chonkie MCP server...", file=sys.stderr)
        logger.info("Starting Chonkie MCP server...")
        print(f"Using Python version: {sys.version}", file=sys.stderr)
        print(f"Script location: {__file__}", file=sys.stderr)

        # Using stdio for Claude Desktop integration
        mcp.run(transport="stdio")
    except Exception as e:
        error_msg = f"Error starting MCP server: {e}\n{traceback.format_exc()}"
        print(error_msg, file=sys.stderr)
        logger.error(error_msg)
        sys.exit(1)