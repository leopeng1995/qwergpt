from typing import List, Dict


class Document:
    def __init__(self, text: str, doc_id: str, metadata: Dict = None):
        self.text = text
        self.doc_id = doc_id
        self.metadata = metadata or {}


class DocumentStore:
    """文档集合管理类"""
    def __init__(self):
        self.documents: Dict[str, Document] = {}

    def add_document(self, document: Document) -> None:
        self.documents[document.doc_id] = document

    def get_document(self, doc_id: str) -> Document:
        return self.documents.get(doc_id)

    def get_all_documents(self) -> List[Document]:
        return list(self.documents.values())


class Chunk:
    def __init__(self, text: str, chunk_id: str, doc_id: str):
        self.text = text
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.embedding = None
