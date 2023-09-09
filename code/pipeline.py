from kedro.pipeline import Pipeline, node, pipeline

from .nodes import loadDocuments


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=loadDocuments,
                inputs=None,
                outputs="documents",
                name="loadDocuments_node",
            ),
        ]
    )
