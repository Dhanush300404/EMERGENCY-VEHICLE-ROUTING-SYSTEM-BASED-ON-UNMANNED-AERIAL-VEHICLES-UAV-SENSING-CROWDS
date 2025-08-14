import os, math, cv2
import osmnx as ox
import networkx as nx
import folium
import requests
import time
import base64
import requests
from telegram import Bot, Update, KeyboardButton, ReplyKeyboardMarkup
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from PIL import Image

GITHUB_USERNAME = "Dhanush300404"
REPO_NAME = "hospital-map"
ACCESS_TOKEN = "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"  # Replace with your actual token

BOT_TOKEN = "7936999111:AAHZAolwTBhkvkthzqx4DWtqniocoH0MWWQ"
bot = Bot(token=BOT_TOKEN)

SHOW = True

# Hospital data
hospitals = [
    {"name": "🏥 Nandha Medical College",     "lat": 11.295497, "lon": 77.626402, "video": 1},
    {"name": "🏥 Thanthai Periyar GH",        "lat": 11.339833, "lon": 77.717301, "video": "v3.mp4"},
    {"name": "🏥 Govt Hospital – Perundurai", "lat": 11.276020, "lon": 77.587407, "video": "v1.mp4"},
]

# YOLOv4-tiny Setup
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
classes = [c.strip() for c in open("coco.names")]
VEHICLE = {"car", "bus", "truck", "motorbike", "motorcycle"}

def detect_traffic(video, title):
    print(f"▶️ Detecting vehicles on: {title}")
    
    # If it's a file path and does not exist, show error
    if isinstance(video, str) and not video.startswith("http") and not os.path.exists(video):
        print("❌ Video not found.")
        return 0

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print("❌ Failed to open video source. Retrying...")
        time.sleep(5)
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            print("❌ Still failed after retry.")
            return 0


    max_count = 0
    start_time = time.time()  # For time-limiting the detection to 12 seconds

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ End of stream or cannot read frame.")
            break

        # Stop after 10 seconds
        if time.time() - start_time > 12:
            print("⏱️ 12 seconds over. Stopping this video.")
            break

        blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True)
        net.setInput(blob)
        outs = net.forward(output_layers)
        count = 0
        for out in outs:
            for det in out:
                scores = det[5:]
                cid = scores.argmax()
                conf = scores[cid]
                if conf > 0.5 and classes[cid] in VEHICLE:
                    count += 1

        max_count = max(max_count, count)

        if SHOW:
            cv2.putText(frame, f"Vehicles: {count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow(title, frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

    cap.release()
    if SHOW:
        cv2.destroyWindow(title)

    print(f"✅ Final max vehicle count for {title}: {max_count}")
    return max_count

def generate_osm_map(user_lat, user_lon, best_hosp):
    print("🗺️ Generating OSM shortest path map...")

    # Step 1: Use OSM + Folium
    G = ox.graph_from_place("Erode, Tamil Nadu, India", network_type='drive')
    source_node = ox.distance.nearest_nodes(G, X=user_lon, Y=user_lat)
    target_node = ox.distance.nearest_nodes(G, X=best_hosp["lon"], Y=best_hosp["lat"])
    route = nx.shortest_path(G, source=source_node, target=target_node, weight='length')
    route_points = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in route]

    m = folium.Map(location=[user_lat, user_lon], zoom_start=13)
    folium.Marker([user_lat, user_lon], popup="📍 Accident", icon=folium.Icon(color='red')).add_to(m)
    folium.Marker([best_hosp["lat"], best_hosp["lon"]], popup=best_hosp["name"], icon=folium.Icon(color='green')).add_to(m)
    folium.PolyLine(route_points, color='blue', weight=5).add_to(m)

    html_file = "shortest_path_map.html"
    image_file = "shortest_path_map.png"
    m.save(html_file)
    print("✅ Map saved as HTML")

    # Step 2: Convert HTML to Image using Selenium
    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--window-size=800,600")
        chrome_options.add_argument("--disable-gpu")

        driver = webdriver.Chrome(options=chrome_options)
        driver.get("file://" + os.path.abspath(html_file))
        time.sleep(3)  # wait for map tiles to load

        driver.save_screenshot(image_file)
        driver.quit()

        # Crop to content (optional, better for cleaner photo)
        img = Image.open(image_file)
        cropped = img.crop((0, 0, 800, 600))  # (left, top, right, bottom)
        cropped.save(image_file)

        print(f"✅ Map screenshot saved as {image_file}")

    except Exception as e:
        print(f"❌ Failed to render image: {e}")
        image_file = None

    return image_file

def upload_to_github(html_file_path):
    url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{REPO_NAME}/contents/shortest_path_map.html"
    headers = {
        "Authorization": f"token {ACCESS_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }

    # First get current file sha (if it exists)
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        sha = response.json()['sha']
    else:
        sha = None

    # Read and encode the new HTML content
    with open(html_file_path, "rb") as f:
        content = base64.b64encode(f.read()).decode()

    data = {
        "message": "Update shortest_path_map.html",
        "content": content,
        "branch": "main",
    }
    if sha:
        data["sha"] = sha

    # Upload (create or update)
    put_response = requests.put(url, json=data, headers=headers)
    if put_response.status_code in [200, 201]:
        print("✅ GitHub upload successful!")
        return f"https://{GITHUB_USERNAME}.github.io/{REPO_NAME}/shortest_path_map.html"
    else:
        print("❌ GitHub upload failed:", put_response.json())
        return None


# Find best hospital based only on OSM shortest distance
def get_best_hospital_osm(user_lat, user_lon):
    G = ox.graph_from_place("Erode, Tamil Nadu, India", network_type='drive')
    source_node = ox.distance.nearest_nodes(G, X=user_lon, Y=user_lat)
    best_hosp = None
    shortest_len = float("inf")

    for hosp in hospitals:
        dest_node = ox.distance.nearest_nodes(G, X=hosp["lon"], Y=hosp["lat"])
        
        try:
            route_len = nx.shortest_path_length(G, source_node, dest_node, weight='length')
            traffic_count = detect_traffic(hosp["video"], hosp["name"])  # 👈 RUN YOLO
            
            print(f"{hosp['name']} → Distance: {route_len/1000:.2f} km, Vehicles: {traffic_count}")
            
            if traffic_count > 20:
                print(f"🚫 Skipping {hosp['name']} due to high traffic ({traffic_count} vehicles).")
                continue

            if route_len < shortest_len:
                shortest_len = route_len
                best_hosp = hosp.copy()
                best_hosp["osm_distance_km"] = route_len / 1000
                best_hosp["vehicles"] = traffic_count

        except Exception as e:
            print(f"⚠️ Skipping {hosp['name']} due to routing error: {e}")
            continue

    return best_hosp


# /start command
def start(update: Update, context: CallbackContext):
    btn = KeyboardButton("📍 Share Location", request_location=True)
    update.message.reply_text(
        "🚑 Send your accident location to find the best hospital (based on shortest road route):",
        reply_markup=ReplyKeyboardMarkup([[btn]], one_time_keyboard=True)
    )

# Handle location input
def handle_location(update: Update, context: CallbackContext):
    loc = update.message.location
    if not loc:
        update.message.reply_text("❌ Location not found. Try again.")
        return

    user_lat, user_lon = loc.latitude, loc.longitude
    best_hosp = get_best_hospital_osm(user_lat, user_lon)
    if best_hosp is None:
        update.message.reply_text("⚠️ No route could be found to any hospital.")
        return

    img_path = generate_osm_map(user_lat, user_lon, best_hosp)
    html_url = upload_to_github("shortest_path_map.html")
    html_upd = "https://dhanush300404.github.io/hospital-map/shortest_path_map.html"

    update.message.reply_text(
        f"✅ Nearest hospital (by road): {best_hosp['name']}\n"
        f"🛣️ Distance: {best_hosp['osm_distance_km']:.2f} km\n\n"
        f"📍 Map saved as `shortest_path_map.html` and sent as image.\n"
        f"🌐 Live updated map: {html_upd if html_upd else '❌ Upload failed'}"
    )
    if img_path:
        bot.send_photo(chat_id=update.effective_chat.id, photo=open(img_path, 'rb'), timeout=120)

    update.message.reply_text("✅ You can send another location anytime using /start.")

# Bot setup
def main():
    updater = Updater(BOT_TOKEN, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(Filters.location, handle_location))
    print("🤖 Bot running. Use /start to begin.")
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()
