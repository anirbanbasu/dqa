from datetime import datetime
import logging
from abc import abstractmethod
from dapr.actor import Actor, ActorInterface, actormethod
from py_a2a_dapr.model.echo_task import EchoInput, EchoResponse, EchoResponseWithHistory


class EchoTaskActorInterface(ActorInterface):
    @abstractmethod
    @actormethod(name="Echo")
    async def echo(self, data: dict | None = None) -> dict | None: ...

    @abstractmethod
    @actormethod(name="History")
    async def history(self) -> list | None: ...

    @abstractmethod
    @actormethod(name="DeleteHistory")
    async def delete_history(self) -> str | None: ...

    @abstractmethod
    @actormethod(name="Cancel")
    async def cancel(self) -> str: ...


logger = logging.getLogger(__name__)


class EchoTaskActor(Actor, EchoTaskActorInterface):
    def __init__(self, ctx, actor_id):
        super().__init__(ctx, actor_id)
        self._cancelled = False
        self._history_key = "echo_history"

    async def on_activate(self) -> None:
        logger.debug(f"{self.__class__.__name__} activated")

    async def on_deactivate(self) -> None:
        logger.debug(f"{self.__class__.__name__} deactivated")

    async def echo(self, data: dict | None = None) -> dict | None:
        if self._cancelled:
            return None
        logger.debug(f"Echo called on actor {self.id} with data: {data}")
        history = await self._state_manager.get_or_add_state(self._history_key, [])
        timestamp = datetime.now()
        input_data = EchoInput.model_validate(data) if data else None
        if not input_data or input_data.user_input.strip() == "":
            echo = f"{EchoTaskActor.__name__}: Hello World! No input text to echo was provided!"
        else:
            echo = f"{EchoTaskActor.__name__}: {input_data.user_input}"

        current = EchoResponse(
            user_input=input_data.user_input if input_data else None,
            output=echo,
            timestamp=timestamp,
            actor_id=str(self.id),
        )
        response = EchoResponseWithHistory(
            current=current,
            past=[EchoResponse.model_validate_json(message) for message in history],
        )
        history.append(current.model_dump_json())
        await self._state_manager.set_state(self._history_key, history)
        await self._state_manager.save_state()
        return response.model_dump()

    async def history(self) -> list | None:
        if self._cancelled:
            return None
        logger.debug(f"History called on actor {self.id}")
        history = await self._state_manager.get_or_add_state(self._history_key, [])
        history_validated = [
            EchoResponse.model_validate_json(message) for message in history
        ]
        response = [item.model_dump() for item in history_validated]
        return response

    async def delete_history(self) -> str | None:
        if self._cancelled:
            return None
        logger.debug(f"DeleteHistory called on actor {self.id}")
        if await self._state_manager.contains_state(self._history_key):
            await self._state_manager.remove_state(self._history_key)
            await self._state_manager.save_state()
            logger.debug(f"History deleted for actor {self.id}")
            return f"History was deleted successfully for {self.id}."
        else:
            logger.debug(f"No history was found to delete for actor {self.id}")
            return f"No history was found for {self.id}."

    async def cancel(self) -> str:
        logger.debug(f"Cancel signal received for actor {self.id}")
        self._cancelled = True
        return "Cancel signal received"
