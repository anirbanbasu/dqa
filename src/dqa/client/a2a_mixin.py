import logging
import httpx


from a2a.client import A2ACardResolver, ClientFactory, ClientConfig
from a2a.types import AgentCard
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH

logger = logging.getLogger(__name__)


class A2AClientMixin:
    async def obtain_a2a_client(
        self,
        httpx_client: httpx.AsyncClient,
        base_url: str,
    ):
        # initialise A2ACardResolver
        resolver = A2ACardResolver(
            httpx_client=httpx_client,
            base_url=base_url,
            # agent_card_path uses default, extended_agent_card_path also uses default
        )
        final_agent_card_to_use: AgentCard | None = None

        logger.info(
            f"Attempting to fetch public agent card from: {base_url}{AGENT_CARD_WELL_KNOWN_PATH}"
        )
        _public_card = (
            await resolver.get_agent_card()
        )  # Fetches from default public path
        logger.info("Successfully fetched public agent card.")
        logger.info(_public_card.model_dump_json(indent=2, exclude_none=True))
        final_agent_card_to_use = _public_card

        client = ClientFactory(
            config=ClientConfig(
                streaming=True, polling=False, httpx_client=httpx_client
            )
        ).create(card=final_agent_card_to_use)
        logger.info("A2A client initialised.")
        return client, final_agent_card_to_use
