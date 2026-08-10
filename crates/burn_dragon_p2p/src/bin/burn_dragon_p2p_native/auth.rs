//! Browser callback handling and native authentication flows.

use super::*;

#[derive(Debug)]
pub(super) enum NativeBrowserAuthCallback {
    ProviderCode {
        provider_code: String,
        state: String,
    },
    AuthResult(Box<NativeCliBridgeAuthResult>),
}

pub(super) struct NativeBrowserAuthListener {
    pub(super) callback_url: String,
    pub(super) nonce: String,
    pub(super) receiver: mpsc::Receiver<Result<NativeBrowserAuthCallback>>,
    pub(super) stop: Arc<AtomicBool>,
    pub(super) join: Option<thread::JoinHandle<()>>,
}

impl NativeBrowserAuthListener {
    pub(super) fn wait(mut self, timeout: Duration) -> Result<NativeBrowserAuthCallback> {
        let callback = match self.receiver.recv_timeout(timeout) {
            Ok(result) => result,
            Err(mpsc::RecvTimeoutError::Timeout) => {
                bail!(
                    "timed out waiting for browser auth callback after {:?}",
                    timeout
                )
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                bail!("browser auth listener terminated before delivering a callback")
            }
        }?;
        self.stop.store(true, Ordering::SeqCst);
        if let Some(join) = self.join.take() {
            join.join().expect("browser auth listener thread");
        }
        Ok(callback)
    }
}

impl Drop for NativeBrowserAuthListener {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

pub(super) fn browser_auth_response_html(title: &str, message: &str) -> String {
    format!(
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>{title}</title></head><body style=\"font-family: ui-monospace, monospace; background: #111; color: #f3f3f3; padding: 2rem;\"><h1 style=\"font-size: 1.1rem; margin-bottom: 1rem;\">{title}</h1><p>{message}</p><script>setTimeout(() => window.close(), 250);</script></body></html>"
    )
}

pub(super) fn write_browser_auth_response(
    stream: &mut TcpStream,
    status: &str,
    body: &str,
) -> Result<()> {
    write!(
        stream,
        concat!(
            "HTTP/1.1 {}\r\n",
            "Content-Type: text/html; charset=utf-8\r\n",
            "Cache-Control: no-store\r\n",
            "Content-Length: {}\r\n",
            "Connection: close\r\n",
            "X-Content-Type-Options: nosniff\r\n",
            "Referrer-Policy: no-referrer\r\n",
            "\r\n{}"
        ),
        status,
        body.len(),
        body,
    )?;
    stream.flush()?;
    Ok(())
}

pub(super) fn parse_native_browser_auth_callback(
    stream: &mut TcpStream,
    expected_nonce: &str,
) -> Result<NativeBrowserAuthCallback> {
    stream.set_read_timeout(Some(NATIVE_AUTH_CALLBACK_READ_TIMEOUT))?;
    let mut reader = BufReader::new(stream.try_clone()?);
    let request_line =
        read_bounded_browser_auth_line(&mut reader, NATIVE_AUTH_CALLBACK_MAX_REQUEST_LINE_BYTES)?
            .ok_or_else(|| anyhow!("browser auth callback closed before request line"))?;
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or_default();
    let target = parts.next().unwrap_or_default();
    if !matches!(method, "GET" | "POST") {
        let _ = write_browser_auth_response(
            stream,
            "405 Method Not Allowed",
            &browser_auth_response_html(
                "login failed",
                "the local auth callback only accepts GET or POST requests.",
            ),
        );
        bail!("browser auth callback used unsupported method {method}");
    }
    let url = Url::parse(&format!("http://127.0.0.1{target}"))
        .with_context(|| format!("failed to parse browser auth callback target {target}"))?;
    if url.path() != "/callback" {
        let _ = write_browser_auth_response(
            stream,
            "404 Not Found",
            &browser_auth_response_html("login failed", "unexpected local callback path."),
        );
        bail!("browser auth callback used unexpected path {}", url.path());
    }

    let mut content_length = 0usize;
    let mut header_bytes = 0usize;
    loop {
        let Some(header) = read_bounded_browser_auth_line(
            &mut reader,
            NATIVE_AUTH_CALLBACK_MAX_HEADER_LINE_BYTES,
        )?
        else {
            break;
        };
        header_bytes = header_bytes
            .checked_add(header.len())
            .ok_or_else(|| anyhow!("browser auth callback headers exceeded maximum size"))?;
        if header_bytes > NATIVE_AUTH_CALLBACK_MAX_HEADER_BYTES {
            bail!(
                "browser auth callback headers exceeded {} bytes",
                NATIVE_AUTH_CALLBACK_MAX_HEADER_BYTES
            );
        }
        if header == "\r\n" || header.is_empty() {
            break;
        }
        if let Some(value) = header.split_once(':')
            && value.0.eq_ignore_ascii_case("content-length")
        {
            content_length = value
                .1
                .trim()
                .parse::<usize>()
                .context("invalid browser auth callback content-length")?;
            if content_length > NATIVE_AUTH_CALLBACK_MAX_BODY_BYTES {
                bail!(
                    "browser auth callback body exceeded {} bytes",
                    NATIVE_AUTH_CALLBACK_MAX_BODY_BYTES
                );
            }
        }
    }

    let mut nonce = None;
    let mut provider_code = None;
    let mut state = None;
    let mut auth_result_json = None;
    let mut error_message = None;
    for (key, value) in url.query_pairs() {
        match key.as_ref() {
            "native_nonce" => nonce = Some(value.into_owned()),
            "provider_code" => provider_code = Some(value.into_owned()),
            "state" => state = Some(value.into_owned()),
            "auth_result_json" => auth_result_json = Some(value.into_owned()),
            "error_message" => error_message = Some(value.into_owned()),
            _ => {}
        }
    }
    if method == "POST" && content_length > 0 {
        let mut body = vec![0_u8; content_length];
        reader.read_exact(&mut body)?;
        for (key, value) in url::form_urlencoded::parse(&body) {
            match key.as_ref() {
                "native_nonce" => nonce = Some(value.into_owned()),
                "provider_code" => provider_code = Some(value.into_owned()),
                "state" => state = Some(value.into_owned()),
                "auth_result_json" => auth_result_json = Some(value.into_owned()),
                "error_message" => error_message = Some(value.into_owned()),
                _ => {}
            }
        }
    }

    if nonce.as_deref() != Some(expected_nonce) {
        let _ = write_browser_auth_response(
            stream,
            "400 Bad Request",
            &browser_auth_response_html("login failed", "the local auth nonce did not match."),
        );
        bail!("browser auth callback nonce mismatch");
    }

    if let Some(message) = error_message.filter(|value| !value.trim().is_empty()) {
        let _ = write_browser_auth_response(
            stream,
            "200 OK",
            &browser_auth_response_html("login failed", &message),
        );
        bail!("browser auth bridge failed: {message}");
    }

    if let Some(auth_result_json) = auth_result_json.filter(|value| !value.trim().is_empty()) {
        let auth_result = serde_json::from_str::<NativeCliBridgeAuthResult>(&auth_result_json)
            .context("failed to decode native auth bridge result")?;
        write_browser_auth_response(
            stream,
            "200 OK",
            &browser_auth_response_html(
                "login complete",
                "GitHub login completed. You can return to the CLI.",
            ),
        )?;
        return Ok(NativeBrowserAuthCallback::AuthResult(Box::new(auth_result)));
    }

    let provider_code = provider_code
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| anyhow!("browser auth callback is missing provider_code"))?;
    let state = state
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| anyhow!("browser auth callback is missing state"))?;
    write_browser_auth_response(
        stream,
        "200 OK",
        &browser_auth_response_html(
            "login complete",
            "GitHub login completed. You can return to the CLI.",
        ),
    )?;
    Ok(NativeBrowserAuthCallback::ProviderCode {
        provider_code,
        state,
    })
}

pub(super) fn read_bounded_browser_auth_line(
    reader: &mut BufReader<TcpStream>,
    max_len: usize,
) -> Result<Option<String>> {
    let mut bytes = Vec::new();
    loop {
        let mut byte = [0_u8; 1];
        match reader.read(&mut byte)? {
            0 => {
                if bytes.is_empty() {
                    return Ok(None);
                }
                break;
            }
            _ => {
                bytes.push(byte[0]);
                if bytes.len() > max_len {
                    bail!("browser auth callback line exceeded {max_len} bytes");
                }
                if byte[0] == b'\n' {
                    break;
                }
            }
        }
    }
    String::from_utf8(bytes)
        .map(Some)
        .context("browser auth callback line was not utf-8")
}

pub(super) fn random_browser_auth_nonce() -> String {
    let mut bytes = [0_u8; 16];
    OsRng.fill_bytes(&mut bytes);
    hex::encode(bytes)
}

pub(super) fn start_native_browser_auth_listener() -> Result<NativeBrowserAuthListener> {
    let listener = TcpListener::bind(("127.0.0.1", 0))
        .context("failed to bind browser auth callback listener")?;
    listener
        .set_nonblocking(true)
        .context("failed to configure browser auth callback listener")?;
    let callback_url = format!(
        "http://127.0.0.1:{}/callback",
        listener.local_addr()?.port()
    );
    let nonce = random_browser_auth_nonce();
    let expected_nonce = nonce.clone();
    let (sender, receiver) = mpsc::channel();
    let stop = Arc::new(AtomicBool::new(false));
    let stop_for_thread = Arc::clone(&stop);
    let join = thread::spawn(move || {
        loop {
            if stop_for_thread.load(Ordering::SeqCst) {
                return;
            }
            match listener.accept() {
                Ok((mut stream, _)) => {
                    let result = parse_native_browser_auth_callback(&mut stream, &expected_nonce);
                    let _ = sender.send(result);
                    stop_for_thread.store(true, Ordering::SeqCst);
                    return;
                }
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    thread::sleep(Duration::from_millis(50));
                }
                Err(error) => {
                    let _ = sender.send(Err(anyhow!(
                        "failed to accept browser auth callback: {error}"
                    )));
                    stop_for_thread.store(true, Ordering::SeqCst);
                    return;
                }
            }
        }
    });
    Ok(NativeBrowserAuthListener {
        callback_url,
        nonce,
        receiver,
        stop,
        join: Some(join),
    })
}

pub(super) fn open_url_in_system_browser(url: &str) -> Result<()> {
    #[cfg(target_os = "macos")]
    {
        let status = Command::new("open").arg(url).status()?;
        if status.success() {
            return Ok(());
        }
        bail!("open exited with status {status}");
    }

    #[cfg(target_os = "windows")]
    {
        let status = Command::new("cmd")
            .args(["/C", "start", "", url])
            .status()?;
        if status.success() {
            return Ok(());
        }
        bail!("start exited with status {status}");
    }

    #[cfg(all(unix, not(target_os = "macos")))]
    {
        for (program, args) in [("xdg-open", vec![url]), ("gio", vec!["open", url])] {
            match Command::new(program).args(args).status() {
                Ok(status) if status.success() => return Ok(()),
                Ok(_) | Err(_) => continue,
            }
        }
        bail!("failed to launch a system browser via xdg-open or gio open");
    }

    #[cfg(not(any(target_os = "macos", target_os = "windows", unix)))]
    {
        bail!("automatic browser launch is not implemented on this platform");
    }
}

pub(super) fn auth_bundle_output_format(path: &Path, format: ConfigFormat) -> Result<OutputFormat> {
    let format = match format {
        ConfigFormat::Auto => infer_format(path)?,
        explicit => explicit,
    };
    match format {
        ConfigFormat::Toml => Ok(OutputFormat::Toml),
        ConfigFormat::Json => Ok(OutputFormat::Json),
        ConfigFormat::Auto => unreachable!(),
    }
}

pub(super) fn write_auth_bundle(
    path: &Path,
    format: ConfigFormat,
    value: &DragonNativeAuthBundle,
) -> Result<()> {
    write_output(Some(path), auth_bundle_output_format(path, format)?, value)
}

pub(super) fn resolve_browser_site_base_url(
    edge_base_url: &str,
    override_base_url: Option<&str>,
) -> Result<String> {
    if let Some(base_url) = override_base_url
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        return Ok(base_url.trim_end_matches('/').to_owned());
    }
    let mut url = Url::parse(edge_base_url)
        .with_context(|| format!("failed to parse edge base URL {edge_base_url}"))?;
    let host = url
        .host_str()
        .ok_or_else(|| anyhow!("edge base URL {edge_base_url} is missing a host"))?
        .to_owned();
    let site_host = host.strip_prefix("edge.").unwrap_or(&host).to_owned();
    url.set_host(Some(&site_host)).map_err(|error| {
        anyhow!("failed to derive browser site host from {edge_base_url}: {error}")
    })?;
    Ok(url.to_string().trim_end_matches('/').to_owned())
}

pub(super) fn infer_browser_site_base_url(edge_base_url: &str) -> Result<String> {
    let override_base_url = env::var(NATIVE_BROWSER_APP_BASE_URL_ENV).ok();
    resolve_browser_site_base_url(edge_base_url, override_base_url.as_deref())
}

pub(super) fn build_native_cli_browser_auth_bootstrap(
    config: &DragonNativePeerConfig,
    _experiment_kind: DragonExperimentKind,
    backend: BackendArg,
    principal_hint: Option<String>,
    session_ttl_secs: i64,
) -> Result<NativeCliBridgeBootstrap> {
    let edge_base_url = config
        .effective_edge_base_url()
        .ok_or_else(|| anyhow!("no edge base URL configured"))?
        .to_owned();
    let site_base_url = infer_browser_site_base_url(&edge_base_url)?;
    let requested_scopes = requested_scopes_for_config(config);
    let (_, identity) = edge_peer_identity_for_storage(config.storage_root.as_path(), None)?;
    Ok(NativeCliBridgeBootstrap {
        edge_base_url: edge_base_url.trim_end_matches('/').to_owned(),
        site_base_url,
        target_artifact_id: native_target_artifact_id(backend).into(),
        app_semver: config.app_semver.to_string(),
        git_commit: config
            .git_commit
            .clone()
            .or_else(build_info::embedded_git_commit_owned)
            .unwrap_or_else(|| "unknown".into()),
        enabled_features_label: config
            .enabled_features_label
            .clone()
            .unwrap_or_else(|| backend.default_enabled_features_label().into()),
        requested_scopes,
        session_ttl_secs,
        principal_hint,
        identity,
    })
}

pub(super) fn build_pending_native_login(
    config: &DragonNativePeerConfig,
    _experiment_kind: DragonExperimentKind,
    backend: BackendArg,
    principal_hint: Option<String>,
    session_ttl_secs: i64,
    use_device_flow: bool,
) -> Result<(tokio::runtime::Runtime, DragonPendingGitHubLogin)> {
    let edge_base_url = config
        .effective_edge_base_url()
        .ok_or_else(|| anyhow!("no edge base URL configured"))?
        .to_owned();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("failed to build async runtime for GitHub login")?;
    let snapshot = runtime.block_on(fetch_edge_snapshot(&edge_base_url))?;
    let release_manifest = native_release_manifest_for_snapshot(config, &snapshot, backend, None)?;
    let requested_scopes = requested_scopes_for_config(config);
    let pending = runtime.block_on(begin_native_github_login(
        &edge_base_url,
        &release_manifest,
        requested_scopes,
        session_ttl_secs,
        principal_hint,
        use_device_flow,
    ))?;
    Ok((runtime, pending))
}

pub(super) fn perform_interactive_native_login(
    config: &DragonNativePeerConfig,
    experiment_kind: DragonExperimentKind,
    backend: BackendArg,
    principal_hint: Option<String>,
    session_ttl_secs: i64,
    callback_timeout_secs: u64,
) -> Result<DragonNativeAuthBundle> {
    let bootstrap = build_native_cli_browser_auth_bootstrap(
        config,
        experiment_kind,
        backend,
        principal_hint.clone(),
        session_ttl_secs,
    )?;
    let listener = start_native_browser_auth_listener()?;
    let bridge_url =
        native_cli_browser_auth_url(&bootstrap, &listener.callback_url, &listener.nonce)?;
    eprintln!("Open this URL if the browser did not open automatically:\n{bridge_url}");
    match open_url_in_system_browser(&bridge_url) {
        Ok(()) => eprintln!("launched browser for GitHub login"),
        Err(error) => {
            eprintln!("automatic browser launch failed: {error}");
        }
    }
    let callback = listener.wait(Duration::from_secs(callback_timeout_secs))?;
    match callback {
        NativeBrowserAuthCallback::AuthResult(result) => {
            let session = finalize_native_auth_session_from_bridge_result(
                &config.storage_root,
                &result,
                None,
            )?;
            Ok(session.auth)
        }
        NativeBrowserAuthCallback::ProviderCode {
            provider_code,
            state,
        } => {
            eprintln!(
                "browser returned provider code only; falling back to native edge completion"
            );
            let (runtime, pending) = build_pending_native_login(
                config,
                experiment_kind,
                backend,
                principal_hint,
                session_ttl_secs,
                false,
            )?;
            if state != pending.login.state {
                bail!("browser auth callback state mismatch");
            }
            let session = runtime.block_on(complete_native_github_login(
                &config.storage_root,
                &pending,
                &provider_code,
                None,
            ))?;
            Ok(session.auth)
        }
    }
}

#[derive(Clone, Debug)]
pub(super) struct NativeAuthResolutionOptions<'a> {
    pub(super) auth_bundle_path: Option<&'a Path>,
    pub(super) auth_bundle_format: ConfigFormat,
    pub(super) principal_hint: Option<String>,
    pub(super) session_ttl_secs: i64,
    pub(super) callback_timeout_secs: u64,
}

pub(super) fn resolve_or_login_native_auth_bundle(
    config: &DragonNativePeerConfig,
    experiment_kind: DragonExperimentKind,
    backend: BackendArg,
    options: NativeAuthResolutionOptions<'_>,
) -> Result<DragonNativeAuthBundle> {
    let mut loaded = if let Some(path) = options.auth_bundle_path {
        if path.is_file() {
            Some(load_typed::<DragonNativeAuthBundle>(
                path,
                options.auth_bundle_format,
            )?)
        } else {
            None
        }
    } else {
        load_cached_native_auth_bundle(&config.storage_root)?
    };

    if let Some(bundle) = loaded.take() {
        if native_auth_bundle_is_fresh(&bundle) {
            if let Some(path) = options.auth_bundle_path {
                write_auth_bundle(path, options.auth_bundle_format, &bundle)?;
            }
            return Ok(bundle);
        }
        match tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .context("failed to build async runtime for auth refresh")?
            .block_on(refresh_native_auth_bundle(
                &config.storage_root,
                &bundle,
                None,
            )) {
            Ok(refreshed) => {
                if let Some(path) = options.auth_bundle_path {
                    write_auth_bundle(path, options.auth_bundle_format, &refreshed)?;
                }
                return Ok(refreshed);
            }
            Err(error) => {
                eprintln!("native auth refresh failed: {error}");
                eprintln!("falling back to interactive browser login");
            }
        }
    }

    let authenticated = perform_interactive_native_login(
        config,
        experiment_kind,
        backend,
        options.principal_hint,
        options.session_ttl_secs,
        options.callback_timeout_secs,
    )?;
    if let Some(path) = options.auth_bundle_path {
        write_auth_bundle(path, options.auth_bundle_format, &authenticated)?;
    }
    Ok(authenticated)
}

pub(super) fn login(args: LoginArgs) -> Result<()> {
    let config = resolved_config(
        args.config.as_deref(),
        args.config_format,
        args.edge_url,
        args.seed_node_urls,
        None,
    )?;
    let auth = perform_interactive_native_login(
        &config,
        args.experiment_kind.into_config(),
        args.backend,
        args.principal_hint,
        args.session_ttl_secs,
        args.callback_timeout_secs,
    )?;
    eprintln!(
        "native auth cache updated: {}",
        default_native_auth_bundle_path(&config.storage_root).display()
    );
    if let Some(path) = args.auth_bundle_out.as_deref() {
        write_auth_bundle(path, ConfigFormat::Auto, &auth)?;
    }
    write_output(None, args.output_format, &auth)
}

pub(super) fn begin_github_login(args: BeginGithubLoginArgs) -> Result<()> {
    let config = resolved_config(
        args.config.as_deref(),
        args.config_format,
        args.edge_url,
        args.seed_node_urls,
        None,
    )?;
    let (_runtime, pending) = build_pending_native_login(
        &config,
        args.experiment_kind.into_config(),
        args.backend,
        args.principal_hint,
        args.session_ttl_secs,
        args.device_flow,
    )?;
    if let Some(authorize_url) = pending.login.authorize_url.as_deref() {
        eprintln!("Open this URL to continue GitHub login:\n{authorize_url}");
    }
    write_output(args.pending_out.as_deref(), args.output_format, &pending)
}

pub(super) fn complete_github_login(args: CompleteGithubLoginArgs) -> Result<()> {
    let config = resolved_config(
        args.config.as_deref(),
        args.config_format,
        None,
        Vec::new(),
        None,
    )?;
    let pending: DragonPendingGitHubLogin = load_typed(&args.pending, args.pending_format)?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("failed to build async runtime for GitHub login completion")?;
    let session = runtime.block_on(complete_native_github_login(
        &config.storage_root,
        &pending,
        &args.provider_code,
        None,
    ))?;
    write_output(
        args.auth_bundle_out.as_deref(),
        args.output_format,
        &session.auth,
    )
}

pub(super) fn enroll_static_principal(args: EnrollStaticPrincipalArgs) -> Result<()> {
    let config = resolved_config(
        args.config.as_deref(),
        args.config_format,
        args.edge_url,
        args.seed_node_urls,
        None,
    )?;
    let edge_base_url = config
        .effective_edge_base_url()
        .ok_or_else(|| anyhow!("no edge base URL configured"))?
        .to_owned();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("failed to build async runtime for static principal enrollment")?;
    let snapshot = runtime.block_on(fetch_edge_snapshot(&edge_base_url))?;
    let release_manifest = native_release_manifest_for_snapshot(
        &config,
        &snapshot,
        args.backend,
        args.target_artifact_hash,
    )?;
    let experiment_id = ExperimentId::new(config.manifest.experiment_id.clone());
    let requested_scopes = match args.principal_kind {
        ManagedPrincipalKindArg::Trainer => managed_trainer_scopes(&experiment_id),
        ManagedPrincipalKindArg::Validator => managed_validator_scopes(&experiment_id),
    };
    let session = runtime.block_on(enroll_native_static_principal(
        &config.storage_root,
        &edge_base_url,
        &release_manifest,
        requested_scopes,
        args.session_ttl_secs,
        args.principal_hint,
        PrincipalId::new(args.principal_id),
        args.trusted_callback_token,
        None,
    ))?;
    write_output(
        args.auth_bundle_out.as_deref(),
        args.output_format,
        &session.auth,
    )
}
