      ### here after frame none
      clone, positive_windows_list = lib.predict_pretrained_ROI(frame, Rectangle(0,0,0,0), netPreMade,False)
      positive_windows[frame_nb]+= positive_windows_list
      new_frame_time = time.time()
      #calculate fps:
      fps = 1/(new_frame_time-prev_frame_time)
      prev_frame_time = new_frame_time 
      fps = int(fps)
      fps_video.append(fps)
      key = cv2.waitKey(1) & 0xFF
      if (clone is not None):
        clone = lib.draw_truth(clone,dict_faces[frame_nb])
        cv2.putText(clone, 'FPS '+str(fps), (2, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)  
        cv2.imshow("frame",clone )
      else:
        frame = lib.draw_truth(frame,dict_faces[frame_nb])
        cv2.putText(frame, 'FPS '+str(fps), (2, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2) 
        cv2.imshow("frame",frame)
    ##########