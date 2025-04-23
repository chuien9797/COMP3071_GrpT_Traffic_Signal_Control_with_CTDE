import xml.etree.ElementTree as ET

net_path = "intersection/double_t/double_t.net.xml"
tree = ET.parse(net_path)
root = tree.getroot()

print("✅ EDGE IDs in double_t.net.xml:")
for edge in root.findall("edge"):
    eid = edge.get("id")
    if eid and not eid.startswith(":"):
        print(eid)
