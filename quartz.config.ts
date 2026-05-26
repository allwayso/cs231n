import { QuartzConfig } from "./quartz/cfg"
import * as Plugin from "./quartz/plugins/loader/plugin-loader"

const config: QuartzConfig = {
  configuration: {
    pageTitle: "CS231n 学习笔记",
    pageTitleSuffix: "",
    enableSPA: true,
    enablePopovers: true,
    analytics: {
      provider: "plausible",
    },
    locale: "zh-CN",
    baseUrl: "allwayso.github.io/cs231n",
    ignorePatterns: ["private", "templates", ".obsidian"],
    theme: {
      fontOrigin: "googleFonts",
      cdnCaching: true,
      typography: {
        header: "Schibsted Grotesk",
        body: "Source Sans Pro",
        code: "IBM Plex Mono",
      },
      colors: {
        lightMode: {
          light: "#fdf6e3",
          lightgray: "#eee8d5",
          gray: "#93a1a1",
          darkgray: "#586e75",
          dark: "#073642",
          secondary: "#268bd2",
          tertiary: "#2aa198",
          highlight: "rgba(101, 123, 131, 0.15)",
          textHighlight: "#b5890055",
        },
        darkMode: {
          light: "#232a2e",
          lightgray: "#2d353b",
          gray: "#9da9a0",
          darkgray: "#d3c6aa",
          dark: "#d3c6aa",
          secondary: "#7fbbb3",
          tertiary: "#83c092",
          highlight: "rgba(84, 58, 72, 0.3)",
          textHighlight: "#543a4855",
        },
      },
    },
  },
  plugins: {
    transformers: [
      Plugin.Community.QuartzTheme({ theme: "everforest", mode: "both" }),
      Plugin.Community.CreatedModifiedDate({ enableDefaultDateType: "modified" }),
      Plugin.Community.SyntaxHighlighting({
        theme: { light: "github-light", dark: "github-dark" },
        keepBackground: false,
      }),
      Plugin.Community.ObsidianFlavoredMarkdown({ enableInHtmlEmbed: false, enableCheckbox: true }),
      Plugin.Community.GitHubFlavoredMarkdown(),
      Plugin.Community.TableOfContents(),
      Plugin.Community.CrawlLinks({ markdownLinkResolution: "shortest" }),
      Plugin.Community.Description(),
      Plugin.Community.Latex({ renderEngine: "katex" }),
      Plugin.Community.HardLineBreaks(),
      Plugin.Community.RemoveDraft(),
      Plugin.Community.AliasRedirects(),
      Plugin.Community.ExplicitPublish(),
      Plugin.Community.UnlistedPages(),
      Plugin.Community.EncryptedPages(),
    ],
    filters: [Plugin.Community.RemoveDraft(), Plugin.Community.ExplicitPublish()],
    emitters: [
      Plugin.Community.AliasRedirects(),
      Plugin.Community.ContentIndex({ enableSiteMap: true, enableRSS: true }),
      Plugin.Community.Favicon(),
      Plugin.Community.OGImage(),
      Plugin.Community.CNAME(),
      Plugin.Community.ContentPage(),
      Plugin.Community.FolderPage(),
      Plugin.Community.TagPage(),
      Plugin.Community.CanvasPage(),
      Plugin.Community.BasesPage(),
    ],
  },
}

export default config