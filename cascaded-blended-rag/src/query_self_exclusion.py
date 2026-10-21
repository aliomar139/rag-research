"""Exclude the query's own document ID before routing/fusion, label-free."""


class QueryContext:
    document_id = None


def without_self(ranking, context, top_k):
    return [(doc, score) for doc, score in ranking if doc != context.document_id][:top_k]


class ExcludingBM25:
    def __init__(self, stage, context):
        self.stage, self.context, self.dfs = stage, context, stage.dfs

    def retrieve_only(self, query, top_k=100):
        return without_self(self.stage.retrieve_only(query, top_k+1), self.context, top_k)


class ExcludingNeural:
    def __init__(self, retriever, context):
        self.retriever, self.context = retriever, context
        self.model = getattr(retriever, "model", None)

    def retrieve(self, query, top_k=100):
        return without_self(self.retriever.retrieve(query, top_k+1), self.context, top_k)
