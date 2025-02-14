
import base64
import zlib
from typing import Optional
import os
import json
from fiddle._src import config
from fiddle._src.experimental import serialization


class ZlibJSONSerializer:
    """Serializer that uses JSON, zlib, and base64 encoding."""

    def serialize(
        self,
        cfg: config.Buildable,
        pyref_policy: Optional[serialization.PyrefPolicy] = None,
    ) -> str:
        
        #return base64.urlsafe_b64encode(
        #    zlib.compress(serialization.dump_json(cfg, pyref_policy).encode())
        #).decode("ascii")
        # print("value:" , serialization.dump_json(cfg, pyref_policy))
        return serialization.dump_json(cfg, pyref_policy)

    def deserialize(
        self,
        serialized: str,
        pyref_policy: Optional[serialization.PyrefPolicy] = None,
    ) -> config.Buildable:
        # print("serialized:", serialized)
        if os.path.isfile(serialized):
            with open(serialized, "r+") as f:
                try:
                    serialized_str = json.load(f)
                except Exception:
                    serialized_str = ""
        else:
            serialized_str = serialized
        # print("serialized_str:", serialized_str)
        return serialization.load_json(serialized_str,pyref_policy=pyref_policy)

