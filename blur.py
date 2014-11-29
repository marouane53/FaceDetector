import numpy as np
import cv2
import os
import pyexiv2

blank = cv2.imread("Blank.JPG")

for root, dirs, files in os.walk('photo/original'):
    for file in files:
        if file.endswith('.jpg'):
            print file

            image = cv2.imread('photo/original/'+file)
            result_image = image.copy()

            # Specify the trained cascade classifier
            face_cascade_name = "haarcascades/haarcascade_frontalface_alt.xml"
            #face_cascade_name = "haarcascades/lbpcascade_profileface.xml"
            #face_cascade_name = "haarcascades/haarcascade_profileface.xml"
            
            # Create a cascade classifier
            face_cascade = cv2.CascadeClassifier()

            # Load the specified classifier
            face_cascade.load(face_cascade_name)

            #Preprocess the image
            grayimg = cv2.cvtColor(image, cv2.cv.CV_BGR2GRAY)
            grayimg = cv2.equalizeHist(grayimg)
            

            #Run the classifiers
            faces = face_cascade.detectMultiScale(grayimg, 1.1, 2, 0|cv2.cv.CV_HAAR_SCALE_IMAGE, (30, 30))
            
			

            

            if len(faces) != 0:         # If there are faces in the images
                print "Faces detected in "+file
                for f in faces:         # For each face in the image

                    # Get the origin co-ordinates and the length and width till where the face extends
                    x, y, w, h = [ v for v in f ]

                    # get the rectangle img around all the faces
                    #cv2.rectangle(image, (x,y), (x+w,y+h), (255,255,0), 5)
                    sub_face = image[y:y+h, x:x+w]
                    #copy faces to /faces
                    face_file_name = "photo/faces/"+ file +  str(y) + ".jpg"
                    cv2.imwrite(face_file_name, sub_face)

                    # apply a gaussian blur on this new recangle image
                    sub_face = cv2.GaussianBlur(sub_face,(23, 23), 30)
                    # merge this blurry rectangle to our final image
                    result_image[y:y+sub_face.shape[0], x:x+sub_face.shape[1]] = sub_face
                    
            
        result_image2 = result_image.copy()
        # Specify the trained cascade classifier
        face_cascade_name2 = "haarcascades/haarcascade_profileface.xml"
        # Create a cascade classifier
        face_cascade2 = cv2.CascadeClassifier()

        # Load the specified classifier
        face_cascade2.load(face_cascade_name2)

        #Preprocess the image
        grayimg2 = cv2.cvtColor(result_image, cv2.cv.CV_BGR2GRAY)
        grayimg2 = cv2.equalizeHist(grayimg2)

        #Run the classifiers
        faces2 = face_cascade2.detectMultiScale(grayimg2, 1.1, 2, 0|cv2.cv.CV_HAAR_SCALE_IMAGE, (30, 30))
        

            

        if len(faces2) != 0:         # If there are faces in the images
            print "Profile Faces detected in "+file
            for f in faces2:         # For each face in the image

                # Get the origin co-ordinates and the length and width till where the face extends
                x, y, w, h = [ v for v in f ]

                # get the rectangle img around all the faces
                #cv2.rectangle(image, (x,y), (x+w,y+h), (255,255,0), 5)
                sub_face2 = result_image[y:y+h, x:x+w]
                
                #copy faces to /faces
                face_file_name2 = "photo/faces/Profile_"+ file +  str(y) + ".jpg"
                cv2.imwrite(face_file_name2, sub_face2)
                

                # apply a gaussian blur on this new recangle image
                sub_face2 = cv2.GaussianBlur(sub_face2,(23, 23), 30)
                # merge this blurry rectangle to our final image
                result_image2[y:y+sub_face2.shape[0], x:x+sub_face2.shape[1]] = sub_face2
            
            
            
            
            
            
            
            
            
        # cv2.imshow("Detected face", result_image)
        cv2.imwrite("photo/blur/"+file, result_image2)
        result_image2 = blank.copy()
        sub_face2     = blank.copy()
        sub_face = blank.copy()
        grayimg = blank.copy()
        image = blank.copy()
        grayimg2 = blank.copy()
        result_image = blank.copy()
        face_file_name2 = blank.copy()
        source_image = pyexiv2.ImageMetadata('photo/original/'+file)
        source_image.read()

        dest_image = pyexiv2.ImageMetadata("photo/blur/"+file)
        dest_image.read()

        source_image.copy(dest_image,exif=True)
        dest_image.write()

            


