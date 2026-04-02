function plot_and_save(results, CQI, dim_scores, dim_names, ...
                       conflict, flags, threshold, best_view, ...
                       final_pred, true_labels, results_dir)
%PLOT_AND_SAVE  Publication-quality figures (PDF/EPS/SVG/EMF).
%
%  v7: Views updated to LDA + NCA + SupervisedTSNE (all supervised).
%
%  Subfigure labels (a)(b)... on all multi-panel figures:
%    Fig1: (a) Accuracy bar   (b) MSE bar
%    Fig5: (a) LDA-BRB  (b) NCA-BRB  (c) TSNE-BRB  (d) Final fusion
%  Single-panel figures (Fig2,3,4,6) have no subfigure label.

if ~exist(results_dir,'dir'), mkdir(results_dir); end

N_VIEWS = numel(results);
VABB    = {'LDA-BRB','NCA-BRB','TSNE-BRB'};
C = [0.2157 0.4941 0.7216;   % blue   — LDA
     0.1725 0.6275 0.1725;   % green  — NCA
     0.8941 0.1020 0.1098;   % red    — SupervisedTSNE
     0.5961 0.3059 0.6392];  % purple — accent
FONT = 'Times New Roman'; FS = 9; LW = 1.2; MS = 5;

    function style_ax(ax)
        set(ax,'FontName',FONT,'FontSize',FS,'Box','on',...
               'GridAlpha',0.3,'GridColor',[.5 .5 .5]);
        grid(ax,'on');
    end

    function add_label(ax, letter)
        % Subfigure label (a)(b)... at top-left of axes (normalised units)
        text(ax, -0.10, 1.06, ['(' letter ')'], ...
             'Units','normalized', ...
             'FontName', FONT, ...
             'FontSize', FS + 2, ...
             'FontWeight', 'bold', ...
             'HorizontalAlignment', 'left', ...
             'VerticalAlignment', 'top', ...
             'Color', [0 0 0]);
    end

    function save_fig(fig, name)
        base = fullfile(results_dir, name);
        set(fig,'PaperPositionMode','auto');
        try; exportgraphics(fig,[base '.pdf'],'ContentType','vector'); catch
             print(fig,base,'-dpdf','-painters'); end
        try; exportgraphics(fig,[base '.eps'],'ContentType','vector'); catch
             print(fig,base,'-depsc2','-painters'); end
        try; print(fig,base,'-dsvg'); catch; end
        try; print(fig,base,'-dmeta'); catch; end
        % MATLAB-editable .fig (open with MATLAB Figure window)
        try; savefig(fig,[base '.fig']); catch; end
        fprintf('  [Saved] %s (.pdf/.eps/.svg/.emf/.fig)\n', name);
    end

%% -----------------------------------------------------------------------
%% Fig 1 — Performance comparison  [(a) Accuracy  (b) MSE]
%% -----------------------------------------------------------------------
acc_v = arrayfun(@(v) results(v).test_acc,         1:N_VIEWS);
mse_v = arrayfun(@(v) results(v).test_mse * 1e3,   1:N_VIEWS);

fig1 = figure('Units','centimeters','Position',[1 1 8 11],'Color','w');

ax1a = subplot(2,1,1,'Parent',fig1);
b1 = bar(ax1a, acc_v, 0.55, 'FaceColor','flat');
for i=1:N_VIEWS, b1.CData(i,:)=C(i,:); end
xticks(ax1a,1:N_VIEWS); xticklabels(ax1a, VABB);
ylabel(ax1a,'Test Accuracy (%)','FontName',FONT,'FontSize',FS);
ylim(ax1a,[max(0,min(acc_v)-5) 102]);
title(ax1a,'Classification Accuracy','FontName',FONT,'FontSize',FS,'FontWeight','bold');
text(ax1a, best_view, acc_v(best_view)+0.8, '(Best)', ...
     'FontName',FONT,'FontSize',7,'HorizontalAlignment','center',...
     'Color',C(4,:),'FontWeight','bold');
style_ax(ax1a);
add_label(ax1a, 'a');

ax1b = subplot(2,1,2,'Parent',fig1);
b2 = bar(ax1b, mse_v, 0.55, 'FaceColor','flat');
for i=1:N_VIEWS, b2.CData(i,:)=C(i,:); end
xticks(ax1b,1:N_VIEWS); xticklabels(ax1b, VABB);
ylabel(ax1b,'Test MSE (\times10^{-3})','FontName',FONT,'FontSize',FS);
title(ax1b,'Mean Squared Error','FontName',FONT,'FontSize',FS,'FontWeight','bold');
style_ax(ax1b);
add_label(ax1b, 'b');

save_fig(fig1,'Fig1_Performance_Comparison');

%% -----------------------------------------------------------------------
%% Fig 2 — CQI composite score  [single panel]
%% -----------------------------------------------------------------------
fig2 = figure('Units','centimeters','Position',[1 1 8 7],'Color','w');
ax2  = axes('Parent',fig2);
b3   = bar(ax2, CQI(:), 0.55, 'FaceColor','flat');
for i=1:N_VIEWS, b3.CData(i,:)=C(i,:); end
hold(ax2,'on');
h_marker = plot(ax2, best_view, CQI(best_view)+0.02, 'v', ...
                'MarkerFaceColor',C(4,:),'MarkerEdgeColor',C(4,:),'MarkerSize',9);
xticks(ax2,1:N_VIEWS); xticklabels(ax2, VABB);
ylabel(ax2,'CQI','FontName',FONT,'FontSize',FS);
ylim(ax2,[0 1.12]);
title(ax2,'Composite Quality Index (CQI): Automatic View Selection','FontName',FONT,'FontSize',FS,'FontWeight','bold');
h_leg = gobjects(N_VIEWS+1,1);
for i=1:N_VIEWS
    h_leg(i) = patch(ax2,NaN,NaN,C(i,:),'EdgeColor','none');
end
h_leg(N_VIEWS+1) = h_marker;
legend(ax2, h_leg, [VABB, {'Selected'}], 'Location','north', ...
       'Orientation','horizontal','FontName',FONT,'FontSize',7,'Box','off');
style_ax(ax2);
save_fig(fig2,'Fig2_CQI_Composite');

%% -----------------------------------------------------------------------
%% Fig 3 — CQI five-dimension breakdown  [single panel]
%%
%%  KEY FIX: dim_scores is N_VIEWS x 5 (3 x 5).
%%  WRONG: bar(dim_scores)   → 3 groups × 5 bars, but xticks/legend expect 5 groups × 3 bars
%%  RIGHT: bar(dim_scores')  → 5 groups (dimensions) × 3 bars (methods) ✓
%% -----------------------------------------------------------------------
fig3 = figure('Units','centimeters','Position',[1 1 14 8],'Color','w');
ax3  = axes('Parent',fig3);
b4   = bar(ax3, dim_scores', 0.72);          % <-- TRANSPOSE: 5 groups x 3 methods
for i=1:N_VIEWS, b4(i).FaceColor=C(i,:); b4(i).EdgeColor='none'; end
xticks(ax3,1:5); xticklabels(ax3, dim_names);
ylabel(ax3,'Dimension Score','FontName',FONT,'FontSize',FS);
ylim(ax3,[0 1.18]);
title(ax3,'CQI Dimension Score Breakdown per View','FontName',FONT,'FontSize',FS,'FontWeight','bold');
legend(ax3, VABB,'Location','northeast','FontName',FONT,'FontSize',FS);
style_ax(ax3);
save_fig(fig3,'Fig3_CQI_Dimensions');

%% -----------------------------------------------------------------------
%% Fig 4 — Conflict entropy distribution  [single panel]
%% -----------------------------------------------------------------------
fig4 = figure('Units','centimeters','Position',[1 1 8 7],'Color','w');
ax4  = axes('Parent',fig4);
n_bins = min(30, max(10, round(sqrt(numel(conflict)))));
conf_conf = conflict(~flags);
conf_unc  = conflict( flags);
h_c = histogram(ax4, conf_conf, n_bins, 'FaceColor',C(1,:), ...
                'EdgeColor','none','FaceAlpha',0.85,'DisplayName','Confident');
hold(ax4,'on');
if ~isempty(conf_unc)
    h_u = histogram(ax4, conf_unc, n_bins, 'FaceColor',C(2,:), ...
                    'EdgeColor','none','FaceAlpha',0.85,'DisplayName','Uncertain');
else
    h_u = plot(ax4,NaN,NaN,'Color',C(2,:),'DisplayName','Uncertain');
end
h_th = xline(ax4, threshold,'--','Color',C(4,:),'LineWidth',1.8,...
             'DisplayName',sprintf('Threshold=%.3f',threshold));
xlabel(ax4,'Conflict Entropy','FontName',FONT,'FontSize',FS);
ylabel(ax4,'Count','FontName',FONT,'FontSize',FS);
title(ax4,'Cross-View Belief Conflict Entropy Distribution',...
      'FontName',FONT,'FontSize',FS,'FontWeight','bold');
legend(ax4,[h_c,h_u,h_th],'Location','northeast','FontName',FONT,'FontSize',FS);
style_ax(ax4);
save_fig(fig4,'Fig4_Conflict_Entropy');

%% -----------------------------------------------------------------------
%% Fig 5 — Predicted vs true scatter
%%   (a) LDA-BRB   (b) NCA-BRB   (c) TSNE-BRB   (d) Multi-View Fusion Final
%% -----------------------------------------------------------------------
LABELS_SET = [0, 0.25, 0.5, 0.75, 1.0];
SUBLABELS  = {'a','b','c','d'};

fig5 = figure('Units','centimeters','Position',[1 1 17 14],'Color','w');

for v = 1:N_VIEWS
    ax = subplot(2,2,v,'Parent',fig5);
    pv = results(v).test_pred(:);
    tv = results(v).true_labels(:);
    scatter(ax, tv, pv, MS^2, C(v,:), 'filled','MarkerFaceAlpha',0.45);
    hold(ax,'on');
    plot(ax,[0 1],[0 1],'k--','LineWidth',LW);
    xlabel(ax,'True Label','FontName',FONT,'FontSize',FS);
    ylabel(ax,'Predicted','FontName',FONT,'FontSize',FS);
    title(ax, VABB{v},'FontName',FONT,'FontSize',FS,'FontWeight','bold');
    xlim(ax,[-0.05 1.05]); ylim(ax,[-0.05 1.05]);
    style_ax(ax);
    text(ax,0.04,0.88,...
         sprintf('MSE=%.4f\nAcc=%.1f%%',mean((pv-tv).^2),results(v).test_acc),...
         'Units','normalized','FontName',FONT,'FontSize',7.5,...
         'BackgroundColor','w','EdgeColor',[.7 .7 .7]);
    add_label(ax, SUBLABELS{v});
end

ax5d = subplot(2,2,4,'Parent',fig5);
ci = ~flags; ui = flags;
h_ci = scatter(ax5d, true_labels(ci), final_pred(ci), MS^2, C(1,:), ...
               'filled','MarkerFaceAlpha',0.55,'DisplayName','Confident');
hold(ax5d,'on');
h_ui = scatter(ax5d, true_labels(ui), final_pred(ui), MS^2, C(2,:), ...
               '^','filled','MarkerFaceAlpha',0.85,'DisplayName','Uncertain');
plot(ax5d,[0 1],[0 1],'k--','LineWidth',LW);
xlabel(ax5d,'True Label','FontName',FONT,'FontSize',FS);
ylabel(ax5d,'Predicted','FontName',FONT,'FontSize',FS);
title(ax5d,'Multi-View Fusion: Uncertainty-Aware Prediction','FontName',FONT,'FontSize',FS,'FontWeight','bold');
xlim(ax5d,[-0.05 1.05]); ylim(ax5d,[-0.05 1.05]);
legend(ax5d,[h_ci,h_ui],'Location','southeast','FontName',FONT,'FontSize',7.5);
style_ax(ax5d);
fin_acc = mean(classify_output(final_pred,LABELS_SET)==true_labels)*100;
text(ax5d,0.04,0.88,...
     sprintf('MSE=%.4f\nAcc=%.1f%%',mean((final_pred-true_labels).^2),fin_acc),...
     'Units','normalized','FontName',FONT,'FontSize',7.5,...
     'BackgroundColor','w','EdgeColor',[.7 .7 .7]);
add_label(ax5d, SUBLABELS{4});

save_fig(fig5,'Fig5_Prediction_Scatter');

%% -----------------------------------------------------------------------
%% Fig 6 — Belief heatmap (best view)  [single panel]
%% -----------------------------------------------------------------------
fig6 = figure('Units','centimeters','Position',[1 1 14 9],'Color','w');
ax6  = axes('Parent',fig6);
B    = results(best_view).belief_out;
tl   = results(best_view).true_labels;
[~,sidx] = sort(tl);
n_disp   = min(200, size(B,1));
didx     = sidx(round(linspace(1,numel(sidx),n_disp)));
imagesc(ax6, B(didx,:)'); colormap(ax6, flipud(bone));
cb = colorbar(ax6);
cb.Label.String='Belief Degree'; cb.Label.FontName=FONT; cb.Label.FontSize=FS;
xlabel(ax6,sprintf('Test Samples sorted by label (n=%d)',n_disp),'FontName',FONT,'FontSize',FS);
ylabel(ax6,'Output Level','FontName',FONT,'FontSize',FS);
yticks(ax6,1:5); yticklabels(ax6,{'0','0.25','0.50','0.75','1.00'});
title(ax6,sprintf('Belief Distribution — %s (CQI Best)',VABB{best_view}),...
      'FontName',FONT,'FontSize',FS,'FontWeight','bold');
style_ax(ax6);
save_fig(fig6,'Fig6_Belief_Heatmap');

close all;
fprintf('  All figures saved to: %s\n', results_dir);
end
