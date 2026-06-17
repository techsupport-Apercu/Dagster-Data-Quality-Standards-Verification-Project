from pathlib import Path
from typing import Any

import pandas as pd
from dagster import IOManager, InputContext, OutputContext, io_manager


class CsvIOManager(IOManager):
    """Dagster IO manager that persists assets as CSV files."""

    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _asset_key_to_filename(self, asset_key_path: list[str]) -> str:
        return "_".join(asset_key_path) + ".csv"

    def _output_filename(self, context: OutputContext) -> str:
        metadata = getattr(context, "definition_metadata", {}) or {}
        return metadata.get("filename", self._asset_key_to_filename(list(context.asset_key.path)))

    def _input_filename(self, context: InputContext) -> str:
        upstream_output = context.upstream_output
        metadata = getattr(upstream_output, "definition_metadata", {}) or {}
        return metadata.get(
            "filename",
            self._asset_key_to_filename(list(upstream_output.asset_key.path)),
        )

    def _path_for(self, filename: str) -> Path:
        return self.base_dir / filename

    def read_csv(self, filename: str) -> pd.DataFrame:
        path = self._path_for(filename)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found at {path}")
        return pd.read_csv(path)

    def write_csv(self, filename: str, dataframe: pd.DataFrame) -> None:
        path = self._path_for(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(path, index=False)

    def handle_output(self, context: OutputContext, obj: Any) -> None:
        if not isinstance(obj, pd.DataFrame):
            raise TypeError(
                f"CsvIOManager only supports pandas.DataFrame outputs, got {type(obj).__name__}."
            )

        filename = self._output_filename(context)
        self.write_csv(filename, obj)
        context.log.info(f"Wrote asset '{context.asset_key.to_string()}' to {self._path_for(filename)}")

    def load_input(self, context: InputContext) -> pd.DataFrame:
        filename = self._input_filename(context)
        dataframe = self.read_csv(filename)
        context.log.info(f"Loaded input from {self._path_for(filename)}")
        return dataframe


@io_manager(config_schema={"base_dir": str})
def csv_io_manager(init_context):
    """Build a CsvIOManager resource from Dagster config."""
    return CsvIOManager(base_dir=init_context.resource_config["base_dir"])
