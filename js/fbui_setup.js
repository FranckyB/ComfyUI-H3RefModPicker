// Shared look for H3 RefModPicker: node colours, the kit stylesheet, and
// graph zoom/drop over our panels. See docs/UI_RULES.md.
import { installAppearance } from "./fbui/appearance.js";
import { installPassthrough } from "./fbui/passthrough.js";
import { ensureKitCss } from "./fbui/theme.js";

installAppearance({ settingsPrefix: "H3RefModPicker", category: "H3 RefMods" });
ensureKitCss();
installPassthrough();
