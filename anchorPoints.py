import cv2
import mediapipe as mp


def findingAnchorPoints(image_path):
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # Initialize FaceMesh with static image mode and maximum 1 face detection
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
    
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image.")
        return

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    # Face landmarks on the image
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Facial landmarks (468 points)
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )

            # Draw landmarks specific to key face features (eyes, lips, etc.)
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )

            #forehead_indices = [9, 10, 107, 338]  # Center forehead and right, left side of forehead
            forehead_indices = [9, 10, 103, 332]
            forehead_landmarks = [(int(face_landmarks.landmark[i].x * image.shape[1]),
                                int(face_landmarks.landmark[i].y * image.shape[0])) for i in forehead_indices]

            #ear_landmarks_indices = [233, 456]  # Top left and right ear points
            ear_landmarks_indices = [139, 389]
            ear_landmarks = [(int(face_landmarks.landmark[i].x * image.shape[1]),
                            int(face_landmarks.landmark[i].y * image.shape[0])) for i in ear_landmarks_indices]
            
            eye_landmarks_indices = [33, 133, 263, 362] #outer left eye, inner left eye, outer right eye, inner right eye
            eye_landmarks = [(int(face_landmarks.landmark[i].x * image.shape[1]),
                            int(face_landmarks.landmark[i].y * image.shape[0])) for i in eye_landmarks_indices]
            
            left_eyebrow_landmarks_indices = [55, 52, 53, 46] #Left edge, Center left, Center right, Right edge
            left_eyebrow_landmarks = [(int(face_landmarks.landmark[i].x * image.shape[1]),
                            int(face_landmarks.landmark[i].y * image.shape[0])) for i in left_eyebrow_landmarks_indices]
            
            right_eyebrow_landmarks_indices = [285, 282, 283, 276] #Left edge, Center left, Center right, Right edge
            right_eyebrow_landmarks = [(int(face_landmarks.landmark[i].x * image.shape[1]),
                            int(face_landmarks.landmark[i].y * image.shape[0])) for i in right_eyebrow_landmarks_indices]
            '''
            anchor_points[0]: center forehead, near hairline
            anchor_points[1]: center forehead, slightly below previous
            anchor_points[2]: left side of forehead, viewer's right
            anchor_points[3]: right side of forehead, viewer's left
            anchor_points[4]: Top of left ear
            anchor_points[5]: Top of right ear
            anchor_points[6]: outer left eye
            anchor_points[7]: inner left eye
            anchor_points[8]: outer right eye
            anchor_points[9]: inner right eye
            anchor_points[10]: Left edge left brow
            anchor_points[11]: Center left left brow
            anchor_points[12]: Center right left brow
            anchor_points[13]: Right edge left brow
            anchor_points[14]: Left edge right brow
            anchor_points[15]: Center left right brow
            anchor_points[16]: Center right right brow
            anchor_points[17]: Right edge right brow
            '''
            anchor_points = forehead_landmarks + ear_landmarks + eye_landmarks + right_eyebrow_landmarks + left_eyebrow_landmarks

    # Resize image to fit the screen, if need be
    height, width, _ = image.shape
    max_height = 800
    max_width = 800

    # Check if resizing is needed
    if height > max_height or width > max_width:
        aspect_ratio = width / height
        new_height = min(max_height, height)
        new_width = int(aspect_ratio * new_height)
        image = cv2.resize(image, (new_width, new_height))

    #comment out the output image of the anchor points
    # cv2.imshow("Facial Landmarks", image)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    return anchor_points