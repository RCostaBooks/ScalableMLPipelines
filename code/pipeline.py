from kedro.pipeline import Pipeline, pipeline, node
from .nodes import 

def create_pipeline(**kwargs) -> Pipeline:
  return pipeline(
    [
      node(
        func=ingestTextFolder,
        inputs=None,
        outputs=vectordb,
        name="ingestDocuments",
        ),
    ]
  )
