import asyncio
import ccxt.pro as ccxt  # Modern CCXT often uses .pro for the async ws extension

async def verify_ws():
    print("Initializing Binance exchange...")
    exchange = ccxt.binance()
    symbol = 'ETH/USDT'
    
    try:
        if exchange.has['watchOrderBook']:
            print(f"Exchange supports watchOrderBook. Attempting to watch {symbol}...")
            # Watch for 5 seconds
            limit = 5
            while limit > 0:
                ob = await exchange.watch_order_book(symbol)
                print(f"Received update: {ob['datetime']} | Bids: {len(ob['bids'])} | Asks: {len(ob['asks'])}")
                limit -= 1
        else:
            print("Exchange does not report 'watchOrderBook' capability.")
            
    except Exception as e:
        print(f"WS Failed: {e}")
    finally:
        await exchange.close()
        print("Exchange closed.")

if __name__ == "__main__":
    try:
        asyncio.run(verify_ws())
    except ImportError:
        print("ccxt.pro not found. Standard ccxt import does not have WebSocket.")
