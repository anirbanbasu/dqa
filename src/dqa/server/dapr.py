from dapr.actor.runtime.runtime import ActorRuntime
from dapr.actor.runtime.config import (
    ActorRuntimeConfig,
    ActorTypeConfig,
    ActorReentrancyConfig,
)
from fastapi import FastAPI
import uvicorn
from dapr.ext.fastapi import DaprActor
from dqa.actor.echo_task import EchoTaskActor
from dqa.actor.mhqa import MHQAActor

from contextlib import asynccontextmanager

from dqa import ParsedEnvVars


@asynccontextmanager
async def lifespan(app: FastAPI):
    dapr_actor = DaprActor(app)
    await dapr_actor.register_actor(EchoTaskActor)
    await dapr_actor.register_actor(MHQAActor)
    yield


app = FastAPI(
    title="DQA Dapr Service",
    # We should be using lifespan instead of on_event
    lifespan=lifespan,
)

config = ActorRuntimeConfig()
config.update_actor_type_configs(
    [
        ActorTypeConfig(
            actor_type=EchoTaskActor.__name__,
            reentrancy=ActorReentrancyConfig(enabled=True),
        ),
        ActorTypeConfig(
            actor_type=MHQAActor.__name__,
            reentrancy=ActorReentrancyConfig(enabled=True),
        ),
    ]
)
ActorRuntime.set_actor_config(config)


def main():
    uvicorn.run(
        app,
        host=ParsedEnvVars().APP_DAPR_SVC_HOST,
        port=ParsedEnvVars().APP_DAPR_SVC_PORT,
    )


if __name__ == "__main__":
    main()
