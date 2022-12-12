from dataclasses import dataclass


@dataclass
class InstanceConf:
    endpoint_url: str
    access_token: str
