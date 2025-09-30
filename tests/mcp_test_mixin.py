from fastmcp import Client


class MCPTestMixin:
    async def list_tools(self, mcp_client: Client):
        """
        Helper method to list tools on the MCP server.
        """
        async with mcp_client:
            result = await mcp_client.list_tools()
            await mcp_client.close()
        return result

    async def list_prompts(self, mcp_client: Client):
        """
        Helper method to list prompts on the MCP server.
        """
        async with mcp_client:
            result = await mcp_client.list_prompts()
            await mcp_client.close()
        return result

    async def list_resources(self, mcp_client: Client):
        """
        Helper method to list resources on the MCP server.
        """
        async with mcp_client:
            result = await mcp_client.list_resources()
            await mcp_client.close()
        return result

    async def call_tool(self, tool_name: str, mcp_client: Client, **kwargs):
        """
        Helper method to call a tool on the MCP server.
        """
        async with mcp_client:
            result = await mcp_client.call_tool(tool_name, arguments=kwargs)
            await mcp_client.close()
        return result

    async def read_resource(self, resource_name: str, mcp_client: Client):
        """
        Helper method to load a resource from the MCP server.
        """
        async with mcp_client:
            result = await mcp_client.read_resource(resource_name)
            await mcp_client.close()
        return result

    async def get_prompt(self, prompt_name: str, mcp_client: Client, **kwargs):
        """
        Helper method to get a prompt from the MCP server.
        """
        async with mcp_client:
            result = await mcp_client.get_prompt(prompt_name, arguments=kwargs)
            await mcp_client.close()
        return result
