Dies ist eine Anleitung zum Nutzen des Workshop_Befehle_Zusammenfassung.ipynb Notebooks:
Um das Notebook nutzen zu können müssen die folgenden Befehle ausgeführt werden:

1. Erstellt eine virtuelle Umgebung und aktiviert sie:

```bash
python3 -m venv venv
```
**Note** Sollte python3 nicht bekannt sein, versucht `/usr/bin/python3 -m venv venv`

2. Installiert die Packages für das Notebook:

```bash
source venv/bin/activate
pip install -r requirements.txt
pip install jupyter python-dotenv
```

3. Startet das Jupyter Notebook:
```bash
jupyter notebook
```

4. Installiert k3d
```bash
curl -s https://raw.githubusercontent.com/k3d-io/k3d/main/install.sh | TAG=v5.4.4 bash
```

5. Erstellt eine lokale Image Registry (erfordert einen laufenden Docker Daemon)
```bash
k3d registry create registry.localhost --port 5000
```

6. Erstellt einen lokalen k8s Cluster
```bash
mkdir -p /tmp/k3dvol

k3d cluster create --volume /tmp/k3dvol:/tmp/k3dvol -p "30081:30081@server:0:direct" -p "6000:6000@server:0:direct" -p "2746:2746@server:0:direct" -p "30084:30084@server:0:direct" --no-lb --k3s-arg '--no-deploy=traefik' --k3s-arg '--no-deploy=servicelb' --registry-use k3d-registry.localhost:5000 sandbox
```

7. Testet, dass der lokale sandbox Cluster funktioniert
```bash
kubectl get nodes
kubectl get namespaces
```
