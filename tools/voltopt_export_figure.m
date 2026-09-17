function voltopt_export_figure(fig, axesList, pdfFile, pngFile, layout)
%VOLTOPT_EXPORT_FIGURE Apply the paper style and export fixed-size PDF/PNG.

if nargin < 5
    layout = 'standard';
end

style = voltopt_plot_style();
axesList = axesList(isgraphics(axesList, 'axes'));

if strcmpi(layout, 'multipanel')
    figureSize = style.MultiPanelFigureSize;
else
    figureSize = style.FigureSize;
end

fig.Units = 'centimeters';
fig.Position(3:4) = figureSize;
fig.Color = 'w';

for k = 1:numel(axesList)
    ax = axesList(k);
    ax.FontName = style.FontName;
    ax.FontSize = style.TickFontSize;
    ax.LineWidth = style.AxesLineWidth;
    ax.TickLabelInterpreter = 'tex';

    style_text_object(ax.XLabel, style.FontName, style.LabelFontSize);
    style_text_object(ax.YLabel, style.FontName, style.LabelFontSize);
    style_text_object(ax.Title, style.FontName, style.TitleFontSize);
    if isprop(ax, 'YAxis') && numel(ax.YAxis) > 1
        for side = 1:numel(ax.YAxis)
            ax.YAxis(side).FontName = style.FontName;
            ax.YAxis(side).FontSize = style.TickFontSize;
            style_text_object( ...
                ax.YAxis(side).Label, style.FontName, style.LabelFontSize);
        end
    end

    plottedLines = findall(ax, 'Type', 'Line');
    for j = 1:numel(plottedLines)
        plottedLines(j).LineWidth = style.PlotLineWidth;
    end
end

legends = findall(fig, 'Type', 'Legend');
for k = 1:numel(legends)
    legends(k).FontName = style.FontName;
    legends(k).FontSize = style.LegendFontSize;
end

colorbars = findall(fig, 'Type', 'ColorBar');
for k = 1:numel(colorbars)
    colorbars(k).FontName = style.FontName;
    colorbars(k).FontSize = style.TickFontSize;
    colorbars(k).LineWidth = style.AxesLineWidth;
end

drawnow;

switch lower(layout)
    case 'standard'
        position_colorbars(colorbars, style.StandardColorbarPosition);
        set_inner_positions(axesList, style.AxesInnerPosition);
    case 'square'
        position_colorbars(colorbars, style.SquareColorbarPosition);
        set_inner_positions(axesList, style.SquareAxesInnerPosition);
    case 'phase'
        set_inner_positions(axesList, style.PhaseAxesInnerPosition);
    case 'multipanel'
        % tiledlayout owns panel positions; its equal tiles and fixed canvas
        % provide the corresponding deterministic multi-panel layout.
    otherwise
        error('Unknown VoltOpt figure layout: %s', layout);
end

drawnow;

% Some MATLAB graphics objects update layout once more during drawnow.
% Reassert the data-axes box after those updates so labels, legends, and
% colorbars cannot change its physical size.
switch lower(layout)
    case 'standard'
        set_inner_positions(axesList, style.AxesInnerPosition);
    case 'square'
        set_inner_positions(axesList, style.SquareAxesInnerPosition);
    case 'phase'
        set_inner_positions(axesList, style.PhaseAxesInnerPosition);
end
drawnow;

fig.PaperUnits = 'centimeters';
fig.PaperPositionMode = 'manual';
fig.PaperPosition = [0.0, 0.0, figureSize];
fig.PaperSize = figureSize;
fig.InvertHardcopy = 'off';
fig.Renderer = 'painters';

if ~isempty(pdfFile)
    print(fig, pdfFile, '-dpdf', '-painters');
end
if ~isempty(pngFile)
    print(fig, pngFile, '-dpng', sprintf('-r%d', style.ExportResolution));
end
end

function style_text_object(obj, fontName, fontSize)
if isgraphics(obj)
    obj.FontName = fontName;
    obj.FontSize = fontSize;
end
end

function set_inner_positions(axesList, position)
for k = 1:numel(axesList)
    axesList(k).Units = 'centimeters';
    axesList(k).PositionConstraint = 'innerposition';
    axesList(k).InnerPosition = position;
end
end

function position_colorbars(colorbars, position)
if numel(colorbars) == 1
    colorbars(1).Units = 'centimeters';
    colorbars(1).Position = position;
end
end
