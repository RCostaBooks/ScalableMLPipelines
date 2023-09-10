from kedro.pipeline import Pipeline, node, pipeline

from .nodes import loadDocuments, textSplitting, createEmbeddings


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=loadDocuments,
                inputs=None,
                outputs="docs",
                name="loadDocuments_node",
            ),
            node(
                func=textSplitting,
                inputs="docs",
                outputs="texts",
                name="textSplitting_node",
            ),
            node(
                func=createEmbeddings,
                inputs="texts",
                outputs=None,
                name="createEmbeddings_node",
            ),
        ]
    )
