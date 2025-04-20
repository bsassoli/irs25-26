from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.schema import Document, TextNode
from typing import List

class HybridMarkdownSentenceParser:
    def __init__(self, chunk_size: int = 256, chunk_overlap: int = 50):
        self.markdown_parser = MarkdownNodeParser()
        self.sentence_splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def get_nodes_from_documents(self, documents: List[Document]) -> List[TextNode]:
        # First, parse the documents using the Markdown parser
        markdown_nodes = self.markdown_parser.get_nodes_from_documents(documents)
        final_nodes = []

        for node in markdown_nodes:
            # If the node's text exceeds the chunk size, split it further
            if len(node.text.split()) > self.sentence_splitter.chunk_size:
                # Create a temporary Document for sentence splitting
                temp_doc = Document(text=node.text)
                # Split the text into smaller chunks
                split_nodes = self.sentence_splitter.get_nodes_from_documents([temp_doc])
                # Preserve metadata from the original node
                for split_node in split_nodes:
                    split_node.metadata.update(node.metadata)
                final_nodes.extend(split_nodes)
            else:
                final_nodes.append(node)

        return final_nodes