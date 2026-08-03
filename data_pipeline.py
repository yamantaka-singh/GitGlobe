import urllib.request
import json
import math
import random

def fetch_github_repos(query="lightweight web server c++", per_page=5):
    """
    Fetches public repositories from GitHub matching a search query.
    """
    url = f"https://api.github.com/search/repositories?q={urllib.request.quote(query)}&per_page={per_page}"
    req = urllib.request.Request(
        url, 
        headers={"User-Agent": "GitGlobe-Pipeline"}
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            return data.get("items", [])
    except Exception as e:
        print(f"Error fetching data: {e}")
        return []

def assign_spherical_coordinates(repos):
    """
    Projects each repository onto a 3D spherical manifold (simulating UMAP output).
    Normalizes coordinates so x^2 + y^2 + z^2 = 1.
    """
    spatial_data = []
    for repo in repos:
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        z = random.uniform(-1, 1)
        
        magnitude = math.sqrt(x**2 + y**2 + z**2)
        if magnitude > 0:
            x, y, z = x / magnitude, y / magnitude, z / magnitude

        repo_node = {
            "name": repo["name"],
            "url": repo["html_url"],
            "description": repo["description"],
            "stars": repo["stargazers_count"],
            "language": repo["language"],
            "coordinates": {"x": round(x, 4), "y": round(y, 4), "z": round(z, 4)}
        }
        spatial_data.append(repo_node)
        
    return spatial_data

if __name__ == "__main__":
    print("🌐 GitGlobe Spatial Pipeline Initialized...\n")
    query_term = "lightweight c++ web server"
    print(f"Searching GitHub repositories for: '{query_term}'...\n")
    
    repos = fetch_github_repos(query_term)
    spatial_nodes = assign_spherical_coordinates(repos)
    
    output_filename = "repos_spatial.json"
    with open(output_filename, "w") as f:
        json.dump(spatial_nodes, f, indent=4)
        
    print(f"Successfully mapped {len(spatial_nodes)} repositories into 3D space!")
    print(f"Data exported to '{output_filename}' successfully.\n")
    
    for node in spatial_nodes:
        c = node["coordinates"]
        print(f"📦 {node['name']} -> Position (X: {c['x']}, Y: {c['y']}, Z: {c['z']})")
