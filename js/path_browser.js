/**
 * H3 RefModPicker's side of the shared file browser (file_browser.js, the
 * same file as in FBnodes): its routes. The browser here only ever shows
 * models/refmods (the backend keeps it there), so there is no Custom folder.
 */

import { api } from "../../scripts/api.js";

export const BROWSER_ROUTES = {
    list: "/h3refmods/path-browser/list",
    file: "/h3refmods/refmod-browser/file",
    remove: "",
    videoFrame: "",
};

// No Custom folder in this pack; the browser is opened with custom: null.
export const getCustomPresetPath = () => "";
export const setCustomPresetPath = async () => {};

/** URL of a RefMod's preview picture (the backend serves only those). */
export function mediaFileUrl(absPath) {
    return api.apiURL(`${BROWSER_ROUTES.file}?path=${encodeURIComponent(absPath)}`);
}
