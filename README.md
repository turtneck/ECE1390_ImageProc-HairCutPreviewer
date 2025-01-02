# ECE1390

### [Google Presentation](https://docs.google.com/presentation/d/1u86NbGBUJEDUaCnIOji0ottfgTjhTsgwbLp0E_kswUI/edit?usp=sharing)

[ECE 1390](https://catalog.upp.pitt.edu/search_advanced.php?cur_cat_oid=225&search_database=Search&search_db=Search&cpage=1&ecpage=1&ppage=1&spage=1&tpage=1&location=3&filter%5Bkeyword%5D=ece+1390&filter%5Bexact_match%5D=1) is a class in the University of Pittsburgh's Electrical and Computer Engineering Department that focus' on image processing. For this course's project, we were tasked with creating an application to utilize image processing techniques from class to create a product. 

1. [Description](#description)
2. [Collaborators](#collaborators)
3. [Code Specifications](#Code_Specifications)
4. [Planned approach](#Planned_approach)
5. [Time-line](#Time-line)
6. [Metrics of Success](#Metrics_of_Success)
7. [Pitfalls and Alternative Solutions](#Pitfalls_and_Alternative_Solutions)

## Description
Too many times individuals have found themselves ending up with a bad haircut. Usually, this is related to the individual not knowing what haircut they wanted, or telling the barber the wrong thing. We built an application called "A Little Off The Top", aimed at helping a user visualize dozens of haircuts before they go to the barbershop or hairstylist. Simple take a picture of yourself, then choose a haircut to put on yourself to visualise what could be you!

The deliverables will be used, in unison with the timeline, to ensure proper software development practices and project engineering. We will utilize a KanBan board to track progress during each stage on issues, and to communicate project updates.

Our target customer is any human ages 9+ that would like to try out a new haircut before going to the barber.

## Collaborators
| Name | GitHub | Module |
| ---------------- | ---------------- | ---------------- |
| Jonah Belback| [Jonah](https://github.com/turtneck) | Face Detector - Bakend |
| John Deibert | [John](https://github.com/jdeibert17) | Anchor Point Calculator - Backend |
| Tim Horrell   | [Tim](https://github.com/tdhorrell)   | Style Fitter - Backend |
| Keshav Shankar   | [Keshav](https://github.com/keshavshankar08)   | Application - Frontend |


## Code Specifications
The goal of the project is to take an image (preferably high-resolution) of the front of the user’s face. 
The software will overlay predefined hair filter data of various hairstyles that the user can try on.
There will be a face detection algorithm to accurately identify the user’s face and its key features (e.g., eyes, nose, forehead, chin) 
so that the hairstyle can be aligned and fitted correctly. The output will be a still image of the user's face with the selected hairstyle applied.
The code should utilize libraries such as OpenCV, Python, NumPy, and other relevant Python libraries that support image processing and facial landmark detection.

## Planned approach
The approach is to
- Take an user accepted photo from a camera
- The User selects a haircut
- Feeding that image into a YOLOv8 Object detection ML Model to detect the face and then export a cropped photo of the face with its bounding-box's cordinates on the orginal image
- Feeding that cropped image to another model that give a bunch fo cordinates of achor points designating different parts of the face like eyebrows, forehead, sides of face, etc.
- Using the anchor points of the face located on the original image to:
  - remove previous hair from photo, repair background
  - add new selected haircut, stretch to fit
- Display this altered image

## Time-line
| Stage | Module| Task | Date Expected |
| --- | --- | --- | --- |
| Prototyping | Face Detector | Container created. Can train and use on image. Can crop to bounding box. | 10/12/2024 |
|  | Anchor Point Calculator | Depict where the anchors are on a face | 10/12/2024 |
|  | Style Fitter | Able to resize style asset to arbitrary anchor points | 10/12/2024 |
|  | Application | Creation of UI to take picture, choose haircut, and show result. | 10/12/2024 |
| Integration | Face Detector | Model can accurately find face | 10/26/2024 |
|  | Anchor Point Calculator | Get anchor points to depict on new faces of given picture | 10/26/2024 |
|  | Style Fitter | Can place resized style asset on top of image at anchor points | 10/26/2024 |
|  | Application | Pass raw picture and style to style fitter. Get edited picture from style fitter. | 10/26/2024 |
| Testing | Face Detector | Improve Model if needed; Fix any bugs | 11/09/2024 |
|  | Anchor Point Calculator | Ensure anchor points are able to be detected on any human face | 11/09/2024 |
|  | Style Fitter | Consider improvements (background smoothing/hair removal) | 11/09/2024 |
|  | Application | Ensure smooth flow of UI interaction. | 11/09/2024 |
| Deploying | Face Detector | Package project into executable for both windows and mac. | 12/09/2024 |
|  | Anchor Point Calculator | Package project into executable for both windows and mac. | 12/09/2024 |
|  | Style Fitter | Package project into executable for both windows and mac. | 12/09/2024 |
|  | Application | Package project into executable for both windows and mac. | 12/09/2024 |

## Metrics of Success
The goal of this project is to build a program that provides real value to the user. The program will be a success if the following metrics are met:
- The user can select at minimum 5 unique haircuts.
- The program fits the desired haircut over the head with correct sizing 80% of the time.
- Edge cases are recognized 80% of the time (multiple faces, dark images, etc.)
- The program will export an image within 10 seconds of running the algorithm.
- The user interface is intuitive (minimal clicks/menus)

## Pitfalls and Alternative Solutions
In developing this program, we expect problems and weak performance in a few key areas. The first is in recognizing invalid images, as garbage input will result in garbage output. For the application to be robust, input image filtering must be done thoroughly. Another area for concern is for faces with varying skin tones and cases where skin tone matches hair color closely. Additionally, after the new haircut is applied, it may look out of place depending on lighting. Lighting correction is not planned for this semester, however is important to consider. 
Should certain modules fail in their functionality, a brute-force method may be implemented. For example, an alternative to face detection and anchoring is requiring photos with a specific head position. This makes applying a haircut more straightforward as well. Should the GUI module fail, all functions will be availble to run from terminal.
