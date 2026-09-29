function style = voltopt_plot_style()
%VOLTOPT_PLOT_STYLE Shared physical layout and typography for paper figures.

style.FontName = 'Times New Roman';
style.TickFontSize = 12;
style.LabelFontSize = 12;
style.LegendFontSize = 12;
style.TitleFontSize = 12;
style.AxesLineWidth = 1.0;
style.PlotLineWidth = 1.5;
style.ExportResolution = 300;

% Physical sizes are in centimeters.  All standard figures share one fixed
% 4:3 data box and the same outer margins.  StandardTightInset is the
% direction-wise maximum measured over all standard paper figures; the
% 0.05 cm safety margin is applied once to the whole class, not per figure.
style.StandardAxesSize = [11.40, 8.55];
style.StandardTightInset = [ ...
    1.82209717569956, 1.67229134632650, ...
    1.61572220203612, 0.621770814803119];
style.StandardOuterInset = style.StandardTightInset + 0.05;
style.AxesInnerPosition = [ ...
    style.StandardOuterInset(1:2), style.StandardAxesSize];
style.FigureSize = [ ...
    style.StandardOuterInset(1) + style.StandardAxesSize(1) + ...
        style.StandardOuterInset(3), ...
    style.StandardOuterInset(2) + style.StandardAxesSize(2) + ...
        style.StandardOuterInset(4)];

% A standard axes with a colorbar forms a separate fixed layout class.
% It uses the same data-box size and left/bottom placement as standard plots.
style.StandardColorbarGap = 0.60;
style.StandardColorbarWidth = 0.35;
style.StandardColorbarRightInset = 1.30;
style.StandardColorbarPosition = [ ...
    style.AxesInnerPosition(1) + style.AxesInnerPosition(3) + ...
        style.StandardColorbarGap, ...
    style.AxesInnerPosition(2), style.StandardColorbarWidth, ...
    style.AxesInnerPosition(4)];
style.StandardColorbarFigureSize = [ ...
    style.StandardColorbarPosition(1) + style.StandardColorbarWidth + ...
        style.StandardColorbarRightInset, ...
    style.FigureSize(2)];

% Square, phase, and multipanel figures are separate fixed layout classes.
style.SquareFigureSize = [16.0, 12.0];
style.SquareAxesInnerPosition = [2.35, 1.85, 8.55, 8.55];
style.SquareColorbarPosition = [11.45, 1.85, 0.35, 8.55];

% The axis-free period plot keeps a square data region on the common canvas.
style.PhaseFigureSize = [16.0, 12.0];
style.PhaseAxesInnerPosition = [2.0, 0.0, 12.0, 12.0];

% The only multi-panel paper figure uses a separate, fixed three-panel canvas.
style.MultiPanelFigureSize = [24.0, 8.5];
end
