from parse_file import parse_semantic_obj
import numpy as np

# Pick a random a point in the triangle
def sample_point_from_triangle(v1, v2, v3):
    r1 = np.random.rand()
    r2 = np.random.rand()
    

    if r1 + r2 > 1:
        r1 = 1 - r1
        r2 = 1 - r2

    return v1 + r1 * (v2 - v1) + r2 * (v3 - v1)
    
def triangle_area(v1, v2, v3):
    return 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))

# For each triangle(face), we get the three vertices and the label of it
# Pick random points from the triangles and label the points
def sample_points_from_mesh(vertices, faces, face_labels, label_to_idx, total_samples = 200000):
    points = []
    labels = []

    valid_faces = []
    valid_labels = []
    face_areas = []

    # for i in range(len(faces)):
       
    #     face = faces[i]
    #     label_str = face_labels[i]
        
    #     if label_str == "<UNK>":
    #         continue
        
    #     label = label_to_idx[label_str]
        
    #     v1 = vertices[face[0]]
    #     v2 = vertices[face[1]]
    #     v3 = vertices[face[2]]

    #     # we sample three random points for each face
    #     samples_per_face = 3
        
    #     for _ in range(samples_per_face): 
    #         p = sample_point_from_triangle(v1, v2, v3)
    #         points.append(p)
    #         labels.append(label)
    # First step is to collect the area of all valid faces, skipping unknown
    for i in range(len(faces)):
        face = faces[i]
        label_str = face_labels[i]

        if label_str == "<UNK>":
            continue

        v1 = vertices[face[0]]
        v2 = vertices[face[1]]
        v3 = vertices[face[2]]

        area = triangle_area(v1,v2,v3)

        valid_faces.append(face)
        valid_labels.append(label_to_idx[label_str])
        face_areas.append(area)
    
    # Computing the total areas
    face_areas = np.array(face_areas, dtype=np.float64)
    total_area = face_areas.sum()

    # Define the number of samples per face proportional to the face area
    expected_samples = face_areas / total_area * total_samples
    samples_per_face = np.floor(expected_samples).astype(int)

    # Distribute remaining samples based on largest fractional parts
    remainder = total_samples - samples_per_face.sum()
    fractional_parts = expected_samples - samples_per_face

    # Distribute the remaining samples to the faces with the largest
    # fractional parts (i.e those nearest to the rounding up)
    if remainder > 0:
        extra_indices = np.argsort(fractional_parts)[-remainder:]
        samples_per_face[extra_indices] += 1

    for i in range(len(valid_faces)):
        face = valid_faces[i]
        label = valid_labels[i]
        num_samples = samples_per_face[i]

        if num_samples == 0:
            continue

        v1 = vertices[face[0]]
        v2 = vertices[face[1]]
        v3 = vertices[face[2]]

        for _ in range(num_samples):
            p = sample_point_from_triangle(v1, v2, v3)
            points.append(p)
            labels.append(label)
            
    return np.array(points), np.array(labels)



def main():
    # obj_path = "area_3/3d/semantic.obj"
    # obj_path = "area_1_no_xyz/area_1/3d/semantic.obj"
    # obj_path = "area_4_no_xyz/area_4/3d/semantic.obj"
    obj_path = "area_6/3d/semantic.obj"
    # obj_path = "area_5b_no_xyz/area_5b/3d/semantic.obj"
    # obj_path = "area_5a_no_xyz/area_5a/3d/semantic.obj"

    vertices, faces, face_labels = parse_semantic_obj(obj_path)

    # build dictionary that maps unique labels to a number
    unique_labels = sorted(set(face_labels))

    label_to_idx = {
    label: i for i, label in enumerate(l for l in unique_labels if l != "<UNK>" )}
    
    
    points, labels = sample_points_from_mesh(vertices, faces, face_labels, label_to_idx)
    print(points[0])
    for la in  label_to_idx:
        print(la, label_to_idx[la])
    key = [keys for keys, val in label_to_idx.items() if val == labels[0]]
    print(str(key))
    print("number of points processed", len(points));

    print("points shape:", points.shape)
    print("labels shape:", labels.shape)
    # print("unique label ids:", np.unique(labels))
    # print("first 5 points:", points[:5])
    # print("first 5 labels:", labels[:5])

    # In the points.npy, each point will be in the format of: 
    # [x_local, y_local, z_local, x_global, y_global, z_global]
    # In the labels.npy there will be lines of labels
    np.save("data/area6/points_area6.npy", points)
    np.save("data/area6/labels_area6.npy", labels)
 
    # percent_imbalance()

# def percent_imbalance():
#     labels = np.load("labels_area4.npy")

#     unique, counts = np.unique(labels, return_counts=True)

#     for u, c in zip(unique, counts):
#         print(f"Class {u}: {c} points ({c / len(labels) * 100:.2f}%)")


    # print(sample[:5])

if __name__ == "__main__":
    main()
    