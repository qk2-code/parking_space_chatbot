import sqlite3
from mcp.server import Server
from mcp.types import Tool, TextContent

app = Server("parking_db_server")

async def get_reservations() -> str:
    """Fetch and return all parking reservations from the database."""
    conn = sqlite3.connect('parking_data.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM reservations")
    rows = cursor.fetchall()
    conn.close()
    if not rows:
        return "No reservations found."
    result = "Reservations:\n" + "\n".join([f"ID: {row[0]}, Customer: {row[1]}, Plate: {row[2]}, Date: {row[3]}, Time: {row[4]}, Duration: {row[5]}" for row in rows])
    return result

@app.list_tools()
async def list_tools():
    return [
        Tool(
            name="get_reservations",
            description="Fetch all parking reservations from the database",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "get_reservations":
        result = await get_reservations()
        return [TextContent(type="text", text=result)]
    raise ValueError(f"Unknown tool: {name}")

if __name__ == "__main__":
    import asyncio
    from mcp.server.stdio import stdio_server

    async def main():
        async with stdio_server() as (read_stream, write_stream):
            await app.run(read_stream, write_stream, app.create_initialization_options())

    asyncio.run(main())
