% Given our abstract representation of a painting,
% render the actual image
%
function img = draw_img(im_rep)

% setup the output data structure
img = ones(im_rep.ymax,im_rep.xmax,3) * 255;

% color table 
% (white,red,yellow,blue,black)
ctable = [255 255 255;
	  255 0 0;
	  255 200 0;
	  0 0 255;
	  0 0 0];

v_pts = im_rep.v_pts;
v_ext = im_rep.v_ext;
v_thick = im_rep.v_thick;

h_pts = im_rep.h_pts;
h_ext = im_rep.h_ext;
h_thick = im_rep.h_thick;

rect = im_rep.rect;
rect_colors = im_rep.rect_colors;

% now draw the image

% first draw rectangles
for r=1:size(rect,1)
  UL = [h_pts(rect(r,3)),v_pts(rect(r,1))];
  LR = [h_pts(rect(r,4)),v_pts(rect(r,2))];
  C = ctable(rect_colors(r),:);
  try
    img = draw_rect(img,UL,LR,C);
  catch ME1
    h_pts
    v_pts
    rect
    pause;    
  end
end

% then draw lines

% horizontal
for hi=1:size(h_ext,1)
  h = h_pts(hi);
  for he=1:(size(h_ext,2)/2)
    he1 = h_ext(hi,2*(he-1)+1);
    he2 = h_ext(hi,2*(he-1)+2);
    if(he1 > 0 && he2 > 0)
      if(h_thick(hi,he) == 0)
	continue;
      end
      v1 = v_pts(he1);
      v2 = v_pts(he2);
      UL = [max(1,h - round(h_thick(hi,he) / 2)), v1];
      LR = [h + round(h_thick(hi,he) / 2), v2];
      C = [0 0 0];
      img = draw_rect(img,UL,LR,C);
    end    
  end
end

% vertical
for vi=1:size(v_ext,1)
  v = v_pts(vi);
  for ve=1:(size(v_ext,2)/2)
    ve1 = v_ext(vi,2*(ve-1)+1);
    ve2 = v_ext(vi,2*(ve-1)+2);
    if(ve1 > 0 && ve2 > 0)
      if(v_thick(vi,ve) == 0)
	continue;
      end
      h1 = h_pts(ve1);
      h2 = h_pts(ve2);
      UL = [h1, max(1,v-round(v_thick(vi,ve) / 2))];
      LR = [h2, v+round(v_thick(vi,ve) / 2)];
      C = [0 0 0];
      img = draw_rect(img,UL,LR,C);
    end    
  end
end

% convert to uint8 for RGB output
img = uint8(img);
