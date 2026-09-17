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

% Physical sizes are in centimeters.  The standard axes box is 4:3 and is
% deliberately independent of labels, legends, and MATLAB's exponent label.
style.FigureSize = [16.0, 12.0];
style.AxesInnerPosition = [2.35, 1.85, 11.40, 8.55];
style.SquareAxesInnerPosition = [2.35, 1.85, 8.55, 8.55];
style.StandardColorbarPosition = [14.35, 1.85, 0.35, 8.55];
style.SquareColorbarPosition = [11.45, 1.85, 0.35, 8.55];

% The axis-free period plot keeps a square data region on the common canvas.
style.PhaseAxesInnerPosition = [2.0, 0.0, 12.0, 12.0];

% The only multi-panel paper figure uses a separate, fixed three-panel canvas.
style.MultiPanelFigureSize = [24.0, 8.5];
end
