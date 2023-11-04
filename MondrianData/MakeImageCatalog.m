%
% Output computer-generated *.eps files to a directory
%
f = figure('Visible','off');

% Load the representations
%
load 'MondriansAndTransatlantics';
for i=1:length(allnames)
  imshow(draw_img(allreps(i)));
  grid off;
  axis off;
  if(labels(i) == 1)
    % Final Neo-plastic Mondrian painting
    saveas(f,sprintf('Mondrian%sCG.eps',allnames{i}),'epsc2');
  else
    % Transatlantic earlier state
    saveas(f,sprintf('Transatlantic%sCG.eps',allnames{i}),'epsc2');
  end
end



