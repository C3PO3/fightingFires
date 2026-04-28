import numpy as np

# obj_path = "area_3/3d/semantic.obj";

def parse_semantic_obj(open_path):
    vertices = []
    faces = []
    face_labels = []

    current_label = None

    with open(open_path, "r") as f:
        for line in f:
            line = line.strip()

            # 1. vertex
            if line.startswith("v "):
                parts = line.split()
                x, y, z = map(float, parts[1:4])
                vertices.append([x, y, z])

            # 2. material label
            elif line.startswith("usemtl "):
                material_name = line.split()[1]
                current_label = material_name.split("_")[0]

            # 3. face
            elif line.startswith("f "):
                parts = line.split()[1:]

                # only take first vertex index
                face = []
                for p in parts:
                    v_idx = int(p.split("/")[0]) - 1 
                    face.append(v_idx)

                if len(face) == 3:
                    faces.append(face)
                    face_labels.append(current_label)

    return np.array(vertices), np.array(faces), face_labels


# vertices, faces, face_labels = parse_semantic_obj(obj_path)

# print("number of vertices:", len(vertices))
# print("number of faces:", len(faces))
# print("number of face labels:", len(face_labels))
# print("first face:", faces[0])
# print("first face label:", face_labels[0])

# face0 = faces[0]
# label0 = face_labels[0]

# v1 = vertices[face0[0]]
# v2 = vertices[face0[1]]
# v3 = vertices[face0[2]]

# print("label:", label0)
# print("triangle indices:", face0)
# print("v1:", v1)
# print("v2:", v2)
# print("v3:", v3)