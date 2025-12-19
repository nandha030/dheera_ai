from mangum import Mangum
from dheera_ai.proxy.proxy_server import app

handler = Mangum(app, lifespan="on")
