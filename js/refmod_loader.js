import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

function findWidget(node, name) {
    return node.widgets?.find((widget) => widget?.name === name) || null;
}

function setWidgetVisible(widget, visible) {
    if (!widget) return;
    if (!widget.__h3LoaderVisibilityWrapped) {
        widget.__h3LoaderVisibilityWrapped = true;
        widget.__h3LoaderBaseComputeSize = widget.computeSize;
        widget.computeSize = function (...args) {
            if (widget.__h3LoaderHidden) {
                return [0, -4];
            }
            if (typeof widget.__h3LoaderBaseComputeSize === "function") {
                return widget.__h3LoaderBaseComputeSize.apply(this, args);
            }
            return [200, Number(LiteGraph?.NODE_WIDGET_HEIGHT || 24)];
        };
    }
    if (typeof widget.__h3LoaderOriginalDisplay === "undefined" && widget.inputEl) {
        widget.__h3LoaderOriginalDisplay = widget.inputEl.style.display;
    }
    widget.__h3LoaderHidden = !visible;
    widget.hidden = !visible;
    if (widget.inputEl) {
        widget.inputEl.style.display = visible ? (widget.__h3LoaderOriginalDisplay || "") : "none";
    }
}

function refreshNodeLayout(node) {
    node.setDirtyCanvas(true, true);
    app.graph?.setDirtyCanvas(true, true);
}

async function fetchLoaderCapabilities(mod) {
    const query = mod ? `?mod=${encodeURIComponent(mod)}` : "";
    const resp = await api.fetchApi(`/h3refmods/refmod-loader/capabilities${query}`);
    if (!resp.ok) {
        throw new Error(`Request failed (${resp.status})`);
    }
    return await resp.json();
}

function applyModalityVisibility(node, caps) {
    const videoWidget = findWidget(node, "video_weight");
    const audioWidget = findWidget(node, "audio_weight");
    const hasVisual = typeof caps?.has_visual === "boolean" ? caps.has_visual : true;
    const hasAudio = typeof caps?.has_audio === "boolean" ? caps.has_audio : true;
    setWidgetVisible(videoWidget, hasVisual);
    setWidgetVisible(audioWidget, hasAudio);
    refreshNodeLayout(node);
}

async function refreshLoaderVisibility(node) {
    const modWidget = findWidget(node, "mod");
    const mod = String(modWidget?.value || "").trim();
    if (!mod) {
        applyModalityVisibility(node, { has_visual: true, has_audio: true });
        return;
    }
    try {
        const data = await fetchLoaderCapabilities(mod);
        applyModalityVisibility(node, data.ok ? data : { has_visual: true, has_audio: true });
    } catch (error) {
        console.warn("[H3RefModLoader] Could not refresh capabilities:", error);
        applyModalityVisibility(node, { has_visual: true, has_audio: true });
    }
}

app.registerExtension({
    name: "H3RefModPicker.RefModLoader",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== "H3RefModLoader") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            const node = this;

            const legacyWeightWidget = findWidget(node, "weight");
            const modWidget = findWidget(node, "mod");
            if (legacyWeightWidget) {
                setWidgetVisible(legacyWeightWidget, false);
            }

            if (modWidget && !modWidget.__h3LoaderRefreshAttached) {
                modWidget.__h3LoaderRefreshAttached = true;
                const originalCallback = modWidget.callback;
                modWidget.callback = function () {
                    const callbackResult = originalCallback?.apply(this, arguments);
                    void refreshLoaderVisibility(node);
                    return callbackResult;
                };
            }

            const onConfigure = node.onConfigure;
            node.onConfigure = function () {
                const configureResult = onConfigure?.apply(this, arguments);
                if (legacyWeightWidget) {
                    setWidgetVisible(legacyWeightWidget, false);
                }
                void refreshLoaderVisibility(this);
                return configureResult;
            };

            void refreshLoaderVisibility(node);
            return result;
        };
    },
});

console.log("[H3RefModPicker] RefModLoader extension loaded");