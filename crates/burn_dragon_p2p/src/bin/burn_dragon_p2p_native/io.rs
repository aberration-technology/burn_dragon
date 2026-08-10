//! Configuration and structured output serialization.

use super::*;

pub(super) fn load_native_config(
    path: &Path,
    format: ConfigFormat,
) -> Result<DragonNativePeerConfig> {
    load_typed(path, format)
}

pub(super) fn load_typed<T>(path: &Path, format: ConfigFormat) -> Result<T>
where
    T: DeserializeOwned,
{
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    let format = match format {
        ConfigFormat::Auto => infer_format(path)?,
        explicit => explicit,
    };
    match format {
        ConfigFormat::Toml => toml::from_str(
            std::str::from_utf8(&bytes)
                .with_context(|| format!("TOML document is not valid UTF-8: {}", path.display()))?,
        )
        .with_context(|| format!("failed to parse TOML {}", path.display())),
        ConfigFormat::Json => serde_json::from_slice(&bytes)
            .with_context(|| format!("failed to parse JSON {}", path.display())),
        ConfigFormat::Auto => unreachable!(),
    }
}

pub(super) fn write_output<T>(path: Option<&Path>, format: OutputFormat, value: &T) -> Result<()>
where
    T: Serialize,
{
    let body = match format {
        OutputFormat::Toml => toml::to_string_pretty(value)?,
        OutputFormat::Json => serde_json::to_string_pretty(value)?,
    };
    if let Some(path) = path {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        fs::write(path, body).with_context(|| format!("failed to write {}", path.display()))?;
    } else {
        println!("{body}");
    }
    Ok(())
}

pub(super) fn infer_format(path: &Path) -> Result<ConfigFormat> {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some("toml") => Ok(ConfigFormat::Toml),
        Some("json") => Ok(ConfigFormat::Json),
        _ => bail!(
            "could not infer config format for {}; pass --config-format",
            path.display()
        ),
    }
}
