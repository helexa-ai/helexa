import React from "react";
import { useTranslation } from "react-i18next";
import { isRtlLanguage, type LanguageCode } from "../i18n/languages";

export type Direction = "forward" | "back";

/**
 * DirectionalIcon
 *
 * Small helper component to render direction-aware icons that respect
 * the current UI writing direction (LTR vs RTL).
 *
 * Usage example:
 *
 *   <DirectionalIcon
 *     direction="forward"
 *     ltrIcon={FaArrowRight}
 *     rtlIcon={FaArrowLeft}
 *   />
 *
 * - `direction="forward"` means “toward the natural reading direction”
 *   (right in LTR, left in RTL).
 * - `direction="back"` means the opposite (left in LTR, right in RTL).
 *
 * You can either:
 *   - pass explicit `ltrIcon` and `rtlIcon` React components, or
 *   - pass a single `icon` component and set `mirrorInRtl` to flip it
 *     horizontally when in RTL (via CSS transform).
 *
 * In most cases, using explicit LTR / RTL icons is clearer and avoids
 * surprises with asymmetric icon shapes.
 */
export interface DirectionalIconProps {
  /**
   * Logical direction relative to reading order.
   * - "forward": in the direction of the text flow
   * - "back": opposite the direction of the text flow
   */
  direction: Direction;

  /**
   * Icon component to use for LTR contexts (e.g. FaArrowRight).
   */
  ltrIcon?: React.ComponentType<{ size?: number | string; className?: string }>;

  /**
   * Icon component to use for RTL contexts (e.g. FaArrowLeft).
   */
  rtlIcon?: React.ComponentType<{ size?: number | string; className?: string }>;

  /**
   * Single base icon component. When provided together with
   * `mirrorInRtl={true}`, it will be mirrored horizontally in RTL.
   */
  icon?: React.ComponentType<{ size?: number | string; className?: string }>;

  /**
   * Whether to flip the `icon` horizontally in RTL.
   * Ignored if both `ltrIcon` and `rtlIcon` are supplied.
   */
  mirrorInRtl?: boolean;

  /**
   * Optional size forwarded to the rendered icon.
   */
  size?: number | string;

  /**
   * Additional className to apply to the rendered icon.
   */
  className?: string;
}

/**
 * Determine if current language is RTL based on i18next language code.
 *
 * Delegates to the shared `isRtlLanguage` helper from i18n/languages.ts
 * so that all RTL logic lives in one place.
 */
const isRtlLanguageCode = (code: string | undefined | null): boolean => {
  if (!code) return false;
  const lang = code.split("-")[0].toLowerCase() as LanguageCode;
  return isRtlLanguage(lang);
};

const DirectionalIcon: React.FC<DirectionalIconProps> = ({
  direction,
  ltrIcon: LtrIcon,
  rtlIcon: RtlIcon,
  icon: BaseIcon,
  mirrorInRtl = false,
  size,
  className,
}) => {
  const { i18n } = useTranslation();
  const isRtl = isRtlLanguageCode(i18n.language);

  // If explicit LTR/RTL icons are provided, prefer those.
  if (LtrIcon && RtlIcon) {
    const IconComponent =
      (direction === "forward" && !isRtl) || (direction === "back" && isRtl)
        ? LtrIcon
        : RtlIcon;

    return <IconComponent size={size} className={className} />;
  }

  // Fallback: single base icon, optionally mirrored in RTL.
  if (!BaseIcon) {
    return null;
  }

  const shouldMirror =
    mirrorInRtl &&
    ((direction === "forward" && isRtl) || (direction === "back" && !isRtl));

  const combinedClassName = [
    className,
    shouldMirror ? "diricon-mirror-rtl" : null,
  ]
    .filter(Boolean)
    .join(" ");

  return <BaseIcon size={size} className={combinedClassName} />;
};

export default DirectionalIcon;
