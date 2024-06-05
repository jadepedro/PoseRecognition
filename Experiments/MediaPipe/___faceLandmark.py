from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt


def draw_landmarks_on_image(rgb_image, detection_result, draw_numbers=True, pause_drawing=800, clear_image=True, landmarklist=[]):
  """
  Draws landmarks on the given RGB image based on the detection result.

  Args:
    rgb_image (numpy.ndarray): The RGB image on which to draw the landmarks.
    detection_result (mediapipe.framework.formats.detection.Detection): The detection result containing the face landmarks.
    draw_numbers (bool, optional): Whether to draw numbers next to the landmarks. Defaults to True.
    pause_drawing (int, optional): The duration (in milliseconds) to pause drawing after each landmark. Defaults to 800.
    clear_image (bool, optional): Whether to clear the image before drawing next landmark. Defaults to True.
    landmarklist (list, optional): A list of landmark indices to draw. If provided, only the landmarks with indices in this list will be drawn. Defaults to an empty list.

  Returns:
    numpy.ndarray: The annotated image with landmarks drawn.
  """
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
      image=annotated_image,
      landmark_list=face_landmarks_proto,
      connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
      landmark_drawing_spec=None,
      connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
    )
    solutions.drawing_utils.draw_landmarks(
      image=annotated_image,
      landmark_list=face_landmarks_proto,
      connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
      landmark_drawing_spec=None,
      connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style()
    )
    solutions.drawing_utils.draw_landmarks(
      image=annotated_image,
      landmark_list=face_landmarks_proto,
      connections=mp.solutions.face_mesh.FACEMESH_IRISES,
      landmark_drawing_spec=None,
      connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style()
    )

    if draw_numbers:
      height, width, _ = annotated_image.shape
      annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
      for i, landmark in enumerate(face_landmarks):
        # if a landmark list has been provided, only draw landmarks in the list
        if landmarklist:
          if i not in landmarklist:
            continue

        copy_image = annotated_image.copy()
        x, y = int(landmark.x * width), int(landmark.y * height)
        cv2.circle(copy_image, (x, y), radius=2, color=(255, 255, 0), thickness=2)
        cv2.putText(copy_image, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        if pause_drawing > 0 and not landmarklist:
          cv2.waitKey(pause_drawing)
        cv2.imshow('Numbered image', copy_image)
        if not clear_image:
          annotated_image = copy_image

  return annotated_image

def plot_face_blendshapes_bar_graph(face_blendshapes):
    
  # Extract the face blendshapes category names and scores.
  face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
  face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
  # The blendshapes are ordered in decreasing score value.
  face_blendshapes_ranks = range(len(face_blendshapes_names))

  fig, ax = plt.subplots(figsize=(12, 12))
  bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
  ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
  ax.invert_yaxis()

  # Label each bar with values
  for score, patch in zip(face_blendshapes_scores, bar.patches):
    plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

  ax.set_xlabel('Score')
  ax.set_title("Face Blendshapes")
  plt.tight_layout()
  plt.show()


# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2


# STEP 2: Create an FaceLandmarker object.
base_options = python.BaseOptions(model_asset_path="face_landmarker.task")
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

# STEP 3: Load the input image.
image = mp.Image.create_from_file("business-person.png")

# STEP 4: Detect face landmarks from the input image.
detection_result = detector.detect(image)

# STEP 5: Process the detection result. In this case, visualize it.
annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
#cv2.imshow('output',cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
plot_face_blendshapes_bar_graph(detection_result.face_blendshapes[0])
#print(detection_result.facial_transformation_matrixes)
cv2.waitKey(0)