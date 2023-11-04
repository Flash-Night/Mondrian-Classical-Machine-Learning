% Draw a rectangle on the image
%
function new_img = draw_rect(img,UL,LR,C)

new_img = img;
for y=UL(1):LR(1)
  for x=UL(2):LR(2)
    try
      new_img(y,x,:) = C;    
    catch
      rethrow(lasterror);
    end
  end
end
