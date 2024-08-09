use std::collections::{HashMap, HashSet};
use std::fs::{create_dir_all, File};
use std::io::{BufRead, BufReader, BufWriter, Read};
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use std::{env, io};

use anyhow::{anyhow, bail, Error, Result};
use async_process::Command;
use async_std::task;
use clap::Parser;
use encoding_rs::WINDOWS_1252;
use encoding_rs_io::DecodeReaderBytesBuilder;
use futures::{future::join_all, join};
use hex::FromHex;
use humansize::{format_size, BINARY};
use md5::{Digest, Md5};
use rayon::prelude::*;
use walkdir::WalkDir;
use bincode::{Decode, Encode};
use directories::ProjectDirs;

// Seemingly optimal buffer size for reading large data on a PCIe 4.0 SSD.
// Need non-blocking queued IO for small files, but is not available in rust.
const READ_BUFFER_SIZE: usize = 128 * 1024;

#[derive(Parser, Debug)]
#[command(version = "1.0.0")]
struct Options {
    /// The workspace to use. If not set, will try to use P4CLIENT. If that is also not set, will try the default one.
    #[arg(short, long)]
    workspace: Option<String>,

    /// The pending changelist to add to. If 0, will add to the default pending changelist.
    #[arg(short, long, default_value = "0")]
    changelist: u32,

    /// Whether we should list file names to stdout. Verbose implies this as well.
    #[arg(short, long)]
    list: bool,

    /// Whether we should output verbose logs to stdout. You should redirect the output to a file if you use this.
    #[arg(short, long)]
    verbose: bool,

    /// If not set, we still do all the work, but don't apply the changes to p4.
    #[arg(short, long)]
    apply: bool,

    /// The files and folders to start from.
    paths: Vec<String>,
}

/// Possible actions of file records in p4 fstat response.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum FileAction {
    Add,
    Edit,
    Delete,
    Branch,
    MoveAdd,
    MoveDelete,
    Integrate,
    Import,
    Purge,
    Archive,
}

// This helps us parse the actions from the p4 fstat response text.
impl std::str::FromStr for FileAction {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        use FileAction::*;
        match s {
            "add" => Ok(Add),
            "edit" => Ok(Edit),
            "delete" => Ok(Delete),
            "branch" => Ok(Branch),
            "move/add" => Ok(MoveAdd),
            "move/delete" => Ok(MoveDelete),
            "integrate" => Ok(Integrate),
            "import" => Ok(Import),
            "purge" => Ok(Purge),
            "archive" => Ok(Archive),
            _ => Err(anyhow!("Invalid file action type \"{}\"", s)),
        }
    }
}

/// Possible types of file records with regards to how we should compute the digest.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum DigestType {
    Binary,
    Text,
    Utf8,
}

// This helps us parse the digest type from the p4 fstat response text.
impl std::str::FromStr for DigestType {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        use DigestType::*;
        if s.starts_with("binary") {
            Ok(Binary)
        } else if s.starts_with("text") {
            Ok(Text)
        } else if s.starts_with("utf8") {
            Ok(Utf8)
        } else if s.starts_with("utf16") {
            // These are mysteriously also using an utf8 digest
            Ok(Utf8)
        } else if s.starts_with("apple") {
            // I have no idea what this means
            Ok(Utf8)
        } else {
            Err(anyhow!("Invalid file digest type \"{}\"", s))
        }
    }
}

/// Information about a single file in the depot, returned by p4 fstat queries.
/// Many of the fields are optional and only appear in specific situations.
#[derive(Default, Debug)]
pub struct DepotFileRecord {
    /// Path in depot syntax, such as "//Depot/Stream/File.ext".
    depot_file: String,

    /// Path in depot syntax, such as "//depot/stream/file.ext".
    depot_file_lower: String,

    /// Path in workspace syntax, such as "C:\Workspace\File.ext" on windows.
    client_file: String,

    /// Path in workspace syntax, such as "c:\workspace\file.ext" on windows.
    client_file_lower: String,

    /// If the file is in the depot, holds the current file type, such as text+w or binary+l.
    head_type: Option<DigestType>,

    /// If the file is in the depot, holds the type of the last change made in the depot.
    /// This tells us if the file existed once but was deleted from the depot.
    head_action: Option<FileAction>,

    /// If the file is in the depot, holds the most recent revision number on the server.
    head_rev: Option<u32>,

    /// If the file is in the workspace, holds the latest revision that we synced.
    /// This may be different from head_rev if we are behind, then the digest does not apply.
    have_rev: Option<u32>,

    /// If the file is in a pending changelist, holds what we are doing with it.
    action: Option<FileAction>,

    /// If the file is in the depot, holds the expected size on disk.
    file_size: Option<u64>,

    /// If the file is in the depot, holds the expected normalized MD5 digest.
    digest: Option<[u8; 16]>,
}

/// Information about the entire depot, returned by fstat queries.
#[derive(Default, Debug)]
pub struct DepotState {
    /// All file records in the depot.
    file_records: Vec<DepotFileRecord>,

    /// Map used to index file_records by depot_file_lower.
    depot_map: HashMap<String, usize>,

    /// Map used to index file_records by client_file_lower.
    client_map: HashMap<String, usize>,
}

impl DepotState {
    pub fn build_mapping(&mut self) {
        self.depot_map.reserve(self.file_records.len());
        self.client_map.reserve(self.file_records.len());

        for (i, record) in self.file_records.iter().enumerate() {
            self.depot_map.insert(record.depot_file_lower.clone(), i);
            self.client_map.insert(record.client_file_lower.clone(), i);
        }
    }

    pub fn has_depot_record(&self, file: &str) -> bool {
        self.depot_map.contains_key(file)
    }

    pub fn get_depot_record(&self, file: &str) -> Option<&DepotFileRecord> {
        match self.depot_map.get(file) {
            Some(i) => Some(&self.file_records[*i]),
            None => None,
        }
    }

    pub fn get_depot_record_mut(&mut self, file: &str) -> Option<&mut DepotFileRecord> {
        match self.depot_map.get(file) {
            Some(i) => Some(&mut self.file_records[*i]),
            None => None,
        }
    }

    pub fn has_client_record(&self, file: &str) -> bool {
        self.client_map.contains_key(file)
    }

    pub fn get_client_record(&self, file: &str) -> Option<&DepotFileRecord> {
        match self.client_map.get(file) {
            Some(i) => Some(&self.file_records[*i]),
            None => None,
        }
    }

    pub fn get_client_record_mut(&mut self, file: &str) -> Option<&mut DepotFileRecord> {
        match self.client_map.get(file) {
            Some(i) => Some(&mut self.file_records[*i]),
            None => None,
        }
    }
}

/// Information about a single file in the workspace.
#[derive(Debug)]
pub struct WorkspaceFile {
    /// Path in workspace syntax, such as C:\Workspace\File.ext on windows.
    path: String,

    /// Path in workspace syntax, such as c:\workspace\file.ext on windows.
    path_lower: String,

    /// The size of the file on disk.
    size: u64,

    /// The modified time of the file on disk.
    date: SystemTime,

    /// Whether this file has been eliminated by one of the ignore filters.
    filtered: bool,
}

// Needed because SystemTime sucks
impl Default for WorkspaceFile {
    fn default() -> Self {
        WorkspaceFile {
            path: String::new(),
            path_lower: String::new(),
            size: 0,
            date: UNIX_EPOCH,
            filtered: false,
        }
    }
}

/// Information about the entire workspace.
#[derive(Default, Debug)]
pub struct WorkspaceState {
    /// All files in the workspace.
    files: Vec<WorkspaceFile>,

    /// Map used to index files by path_lower.
    file_map: HashMap<String, usize>,

    // The number of files not filtered.
    num_files: usize,
}

/// Used to store cached digest for a file.
#[derive(Debug, Encode, Decode)]
pub struct WorkspaceCacheEntry {
    /// File size during last run.
    size: u64,

    /// Modified date during last run.
    date: SystemTime,

    /// Digest during last run.
    digest: [u8; 16],
}

// Needed because SystemTime sucks
impl Default for WorkspaceCacheEntry {
    fn default() -> Self {
        WorkspaceCacheEntry {
            size: 0,
            date: UNIX_EPOCH,
            digest: [0; 16],
        }
    }
}

/// Used to store cached digests for a workspace.
#[derive(Default, Debug, Encode, Decode)]
pub struct WorkspaceCache {
    /// Map used to index files by path_lower.
    file_map: HashMap<String, WorkspaceCacheEntry>,

    /// Whether the cache is out of date.
    out_of_date: bool,
}

impl WorkspaceState {
    pub fn build_mapping(&mut self) {
        self.file_map.reserve(self.files.len());

        for (i, file) in self.files.iter().enumerate() {
            self.file_map.insert(file.path_lower.clone(), i);
        }
    }

    pub fn has_file(&self, file: &str) -> bool {
        self.file_map.contains_key(file)
    }

    pub fn has_filtered(&self, file: &str) -> bool {
        match self.file_map.get(file) {
            Some(i) => !self.files[*i].filtered,
            None => false,
        }
    }

    pub fn get_file(&self, file: &str) -> Option<&WorkspaceFile> {
        match self.file_map.get(file) {
            Some(i) => Some(&self.files[*i]),
            None => None,
        }
    }

    pub fn get_filtered(&self, file: &str) -> Option<&WorkspaceFile> {
        match self.file_map.get(file) {
            Some(i) => {
                let file = &self.files[*i];
                if file.filtered {
                    None
                } else {
                    Some(file)
                }
            }
            None => None,
        }
    }
}

/// Runs one slice of a batched p4 command (eg. p4 stuff a100 a101 ... a198 a199)
async fn run_p4_command_slice(
    options: &Options,
    work_dir: &String,
    always_args: &[&'static str],
    batched_args_slice: &[String],
    use_changelist: bool,
) -> Result<Vec<String>> {
    let mut cmd = Command::new("p4");
    cmd.current_dir(work_dir);

    if options.workspace.is_some() {
        cmd.arg("-c");
        cmd.arg(&options.workspace.as_ref().unwrap());
    }

    cmd.args(always_args);

    if use_changelist && options.changelist != 0 {
        cmd.arg("-c");
        cmd.arg(options.changelist.to_string());
    }

    cmd.args(batched_args_slice);

    let output = cmd.output().await?;
    let data = &output.stdout[..];

    let decoder = DecodeReaderBytesBuilder::new()
        .encoding(Some(WINDOWS_1252))
        .build(data);

    let mut lines = BufReader::new(decoder).lines();
    let mut result = Vec::new();

    loop {
        match lines.next() {
            Some(Ok(line)) => {
                result.push(line.to_string());
            }
            Some(Err(e)) => return Err(e.into()),
            None => break,
        }
    }

    Ok(result)
}

/// Runs a p4 command with thousands of arguments in multiple batches to bypass windows input limit
async fn run_p4_command_batched(
    options: &Options,
    work_dir: &String,
    always_args: &[&'static str],
    batched_args: &[String],
    use_changelist: bool,
) -> Result<Vec<String>> {
    let mut futures = Vec::new();
    let mut batch_start = 0;
    let mut batch_end = 0;
    let mut batch_size = 0;
    let mut batch_count = 0;

    // Borrow check wouldn't let me access these in the tasks without cloning.
    // But that's not actually necessary because we're awaiting all tasks before exiting this function.
    let unsafe_static_options =
        unsafe { std::mem::transmute::<&Options, &'static Options>(&options) };
    let unsafe_static_work_dir =
        unsafe { std::mem::transmute::<&String, &'static String>(&work_dir) };
    let unsafe_static_always_args =
        unsafe { std::mem::transmute::<&[&'static str], &'static [&'static str]>(&always_args) };
    let unsafe_static_batched_args =
        unsafe { std::mem::transmute::<&[String], &'static [String]>(&batched_args) };

    // Start full batches
    for arg in batched_args {
        let arg_size = arg.len();

        if batch_size + arg_size > 32000 {
            futures.push(task::spawn(run_p4_command_slice(
                &unsafe_static_options,
                &unsafe_static_work_dir,
                &unsafe_static_always_args,
                &unsafe_static_batched_args[batch_start..batch_end],
                use_changelist,
            )));

            batch_count = batch_count + 1;

            batch_start = batch_end;
            batch_size = 0;
        }

        batch_size = batch_size + arg_size + 1;
        batch_end = batch_end + 1;
    }

    // Start final, less full batch
    if batch_size > 0 {
        futures.push(task::spawn(run_p4_command_slice(
            &unsafe_static_options,
            &unsafe_static_work_dir,
            &unsafe_static_always_args,
            &unsafe_static_batched_args[batch_start..batch_end],
            use_changelist,
        )));

        batch_count = batch_count + 1;
    }

    println!("      Running \"p4 {}\" with {} batches.", always_args[0], batch_count);

    // Merge batches into single output
    let batched_output = join_all(futures.into_iter()).await;
    let mut results = Vec::new();

    for maybe_batch in batched_output {
        match maybe_batch {
            Ok(lines) => results.extend(lines),
            Err(e) => return Err(e.into()),
        }
    }

    Ok(results)
}

async fn parse_p4_fstat_response(output: Vec<String>) -> Result<Vec<DepotFileRecord>> {
    let mut records: Vec<DepotFileRecord> = Vec::new();
    let mut pending_record: DepotFileRecord = Default::default();

    for line in output {
        if line.len() > 3 {
            // This is more information for the current record
            let mut split = line[4..].splitn(2, ' ');
            let key = split
                .next()
                .ok_or(anyhow!("Invalid key in fstat response"))?;
            let value = split
                .next()
                .ok_or(anyhow!("Invalid value in fstat response"))?;
            match key {
                "depotFile" => {
                    pending_record.depot_file = value.to_string();
                    pending_record.depot_file_lower =
                        pending_record.depot_file.to_ascii_lowercase();
                }
                "clientFile" => {
                    pending_record.client_file = value.to_string();
                    pending_record.client_file_lower =
                        pending_record.client_file.to_ascii_lowercase();
                }
                "headType" => pending_record.head_type = Some(value.parse()?),
                "headRev" => pending_record.head_rev = Some(value.parse()?),
                "haveRev" => pending_record.have_rev = Some(value.parse()?),
                "headAction" => pending_record.head_action = Some(value.parse()?),
                "action" => pending_record.action = Some(value.parse()?),
                "digest" => pending_record.digest = Some(<[u8; 16]>::from_hex(value)?),
                "fileSize" => pending_record.file_size = Some(value.parse()?),
                _ => (),
            }
        } else {
            // Record finished on empty line, commit and reset for next
            records.push(pending_record);
            pending_record = Default::default();
        }
    }

    Ok(records)
}

async fn run_p4_fstat_all(options: &Options, work_dir: &String) -> Result<DepotState> {
    println!("   Requesting depot state for all files.");
    let start_time = Instant::now();

    let fstat_args = [
        "fstat",
        "-Rc", // Only files mapped into current workspace
        "-Ol", // Include file size and digest for files in depot
        "-T depotFile clientFile headAction headType headRev haveRev digest fileSize action",
    ];

    // First get latest depot state for everything in one slice
    let initial_args = [String::from("./...")];
    let response =
        run_p4_command_batched(&options, &work_dir, &fstat_args, &initial_args, false).await?;

    if options.verbose {
        println!("         BEGIN FSTAT RESPONSE DUMP");
        for line in &response {
            println!("{}", &line);
        }
        println!("         END FSTAT RESPONSE DUMP");
    }

    let mut depot_state: DepotState = Default::default();
    depot_state.file_records = parse_p4_fstat_response(response).await?;

    // Build hashmaps to find records
    depot_state.build_mapping();

    // Find records with out of date revisions
    let mut old_records = Vec::new();
    for record in &depot_state.file_records {
        if record.have_rev.is_some() && record.head_rev.is_some() {
            let head_rev = record.head_rev.unwrap();
            let have_rev = record.have_rev.unwrap();

            if head_rev != have_rev {
                old_records.push(
                    [
                        record.depot_file.clone(),
                        record.have_rev.unwrap().to_string(),
                    ]
                    .join("#"),
                );

                if options.verbose {
                    println!(
                        "         File \"{}\" is outdated (head rev {}, have rev {}",
                        &record.depot_file, head_rev, have_rev
                    );
                }
            }
        }
    }

    println!(
        "      Received {} fstat records in {} seconds.",
        depot_state.file_records.len(),
        start_time.elapsed().as_secs_f32()
    );

    // Request new records for them and update the digests
    if old_records.len() > 0 {
        println!(
            "   Requesting depot state for {} outdated files.",
            old_records.len()
        );

        let start_time = Instant::now();

        let response =
            run_p4_command_batched(&options, &work_dir, &fstat_args, &old_records, false).await?;

        if options.verbose {
            println!("         BEGIN FSTAT RESPONSE DUMP");
            for line in &response {
                println!("{}", &line);
            }
            println!("         END FSTAT RESPONSE DUMP");
        }

        let refreshed_records = parse_p4_fstat_response(response).await?;

        for refreshed_record in &refreshed_records {
            match depot_state.get_depot_record_mut(&refreshed_record.depot_file_lower) {
                Some(original_record) => {
                    original_record.head_type = refreshed_record.head_type;
                    original_record.head_action = refreshed_record.head_action;
                    original_record.file_size = refreshed_record.file_size;
                    original_record.digest = refreshed_record.digest;

                    if options.verbose {
                        println!(
                            "         Updated record for \"{}\"",
                            original_record.depot_file
                        );
                    }
                }
                None => {
                    bail!(
                        "Failed to find original record for \"{}\"",
                        &refreshed_record.depot_file
                    )
                }
            }
        }

        println!(
            "      Updated {} fstat records in {} seconds.",
            refreshed_records.len(),
            start_time.elapsed().as_secs_f32()
        );
    }

    Ok(depot_state)
}

/// Scans workspace for files that are not ignored.
async fn gather_workspace(options: &Options, work_dir: &String) -> Result<WorkspaceState> {
    println!("   Scanning workspace for files.");
    let start_time = Instant::now();

    // Gather files
    let mut workspace_state: WorkspaceState = Default::default();
    let mut num_dirs = 0;
    let mut total_size = 0;

    for entry in WalkDir::new(&work_dir) {
        match entry {
            Ok(path) => {
                if path.file_type().is_file() {
                    let path_string = path.path().display().to_string();
                    let meta = path.metadata()?;
                    total_size = total_size + meta.len();

                    workspace_state.files.push(WorkspaceFile {
                        path_lower: path_string.to_ascii_lowercase(),
                        path: path_string,
                        size: meta.len(),
                        date: meta.modified()?,
                        filtered: false,
                    });
                } else if path.file_type().is_dir() {
                    num_dirs = num_dirs + 1;
                }
            }
            Err(e) => return Err(e.into()),
        }
    }

    workspace_state.files.sort_by(|a, b| a.path_lower.cmp(&b.path_lower));

    if options.verbose {
        for file in &workspace_state.files {
            println!("         File \"{}\", size {}", file.path, file.size);
        }
    }

    println!(
        "      Collected {} files in {} directories ({}) in {} seconds.",
        workspace_state.files.len(),
        num_dirs,
        format_size(total_size, BINARY),
        start_time.elapsed().as_secs_f32()
    );

    println!("   Filtering workspace files.");
    let start_time = Instant::now();

    // Filter by p4 ignores
    let ignores_args = ["ignores", "-i"];
    let ignores_paths = workspace_state
        .files
        .iter()
        .map(|f| f.path.clone())
        .collect::<Vec<_>>();

    let ignored_files: Vec<String> =
        run_p4_command_batched(&options, &work_dir, &ignores_args, &ignores_paths, false).await?;

    if options.verbose {
        for ignored_file in &ignored_files {
            println!("         Ignored file \"{}\" by ignores", ignored_file);
        }
    }

    let mut ignored_count = 0;
    let mut ignored_files_hash: HashSet<String> = HashSet::new();

    for file in ignored_files {
        ignored_files_hash.insert(file[..file.len() - 8].to_string());
    }

    for file in workspace_state.files.iter_mut() {
        if ignored_files_hash.contains(&file.path) {
            ignored_count = ignored_count + 1;
            file.filtered = true;
        }
    }

    // TODO: Not needed? Same count with it disabled, what was this for?
    /*
    // Filter by p4 where
    let where_args = ["-F", "%mapFlag%%localPath%", "where"];
    let where_paths = workspace_state
        .files
        .iter()
        .filter(|f| !f.filtered)
        .map(|f| f.path.clone())
        .collect::<Vec<_>>();

    let where_files: Vec<String> =
        run_p4_command_batched(&options, &work_dir, &where_args, &where_paths, false).await?;

    let mut where_files_hash: HashSet<String> = HashSet::new();

    for file in where_files {
        if file.starts_with("-") {
            where_files_hash.insert(file[1..].to_string());

            if options.verbose {
                println!("         Ignored file \"{}\" by where", file);
            }
        } else {
            where_files_hash.remove(&file);
        }
    }

    for file in workspace_state.files.iter_mut() {
        if !file.filtered && where_files_hash.contains(&file.path) {
            ignored_count = ignored_count + 1;
            file.filtered = true;
        }
    }
    */

    // Build hashmap
    workspace_state.build_mapping();
    workspace_state.num_files = workspace_state.files.len() - ignored_count;

    println!(
        "      Filtered out {} files, {} remain, in {} seconds.",
        ignored_count,
        workspace_state.num_files,
        start_time.elapsed().as_secs_f32()
    );

    Ok(workspace_state)
}

/// Computes the digest for a binary file, simple MD5.
fn compute_digest_binary(file: &WorkspaceFile, hasher: &mut Md5) -> Result<()> {
    let mut file = File::open(&file.path).expect("Failed to open file for hash.");
    let mut buffer = [0; READ_BUFFER_SIZE];

    loop {
        match file.read(&mut buffer) {
            Ok(0) => return Ok(()),
            Ok(len) => hasher.update(&buffer[..len]),
            Err(ref e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e.into()),
        }
    }
}

/// Computes the digest for a text buffer, normalized line endings MD5.
fn update_text_digest_utf8<R: BufRead + Read>(
    input: &mut R,
    line_buffer: &mut Vec<u8>,
    hasher: &mut Md5,
) -> Result<()> {
    loop {
        match input.read_until(b'\n', line_buffer) {
            Ok(0) => return Ok(()),
            Ok(_n) => {
                if line_buffer.ends_with(b"\r\n") {
                    line_buffer.remove(line_buffer.len() - 2);
                }
                md5::digest::Update::update(hasher, &line_buffer);
                line_buffer.clear();
            }
            Err(e) => return Err(e.into()),
        }
    }
}

/// Computes the digest for a text file, normalized line endings MD5.
fn compute_digest_text(file: &WorkspaceFile, hasher: &mut Md5) -> Result<()> {
    let file = File::open(&file.path).expect("Failed to open file for hash.");
    let mut read = BufReader::with_capacity(READ_BUFFER_SIZE, file);
    let mut line_buffer = Vec::new();

    update_text_digest_utf8(&mut read, &mut line_buffer, hasher)
}

/// Computes the digest for a utf8 file, normalized line endings MD5 without BOM.
/// This function is a bit slower, but only few files have this encoding, so it is fine.
fn compute_digest_utf8(file: &WorkspaceFile, hasher: &mut Md5) -> Result<()> {
    let file = File::open(&file.path).expect("Failed to open file for hash.");
    let mut buffer = [0u8; READ_BUFFER_SIZE];
    let mut line_buffer = Vec::new();
    let mut full_buffer = Vec::new();

    let len = DecodeReaderBytesBuilder::new()
        .utf8_passthru(false)
        .bom_sniffing(true)
        .strip_bom(true)
        .build_with_buffer(file, &mut buffer[..])?
        .read_to_end(&mut full_buffer)?;

    let mut filled = &full_buffer[..len];
    update_text_digest_utf8(&mut filled, &mut line_buffer, hasher)
}

/// Computes digests for a number of files in the workspace.
fn parallel_compute_digests<'a>(
    files: Vec<(&'a WorkspaceFile, DigestType)>,
    cache: &mut WorkspaceCache,
) -> Vec<(&'a WorkspaceFile, [u8; 16], bool)> {
    let results : Vec<(&'a WorkspaceFile, [u8; 16], bool)> = files
        .into_par_iter()
        .with_max_len(1)
        .map(|file| {
            // Check cache first
            if let Some(cache_entry) = cache.file_map.get(&file.0.path_lower) {
                if cache_entry.size == file.0.size && cache_entry.date == file.0.date {
                    return (file.0, cache_entry.digest, true);
                }
            }

            let mut hasher = Md5::new();
            let mut digest: [u8; 16] = Default::default();

            match file.1 {
                DigestType::Binary => compute_digest_binary(file.0, &mut hasher).unwrap(),
                DigestType::Text => compute_digest_text(file.0, &mut hasher).unwrap(),
                DigestType::Utf8 => compute_digest_utf8(file.0, &mut hasher).unwrap(),
            }

            digest.copy_from_slice(&hasher.finalize()[..16]);
            (file.0, digest, false)
        })
        .collect();

    // Update cache
    for result in &results {
        if !result.2 {
            cache.file_map.insert(
                result.0.path_lower.clone(),
                WorkspaceCacheEntry {
                    size: result.0.size,
                    date: result.0.date,
                    digest: result.1,
                },
            );
            cache.out_of_date = true;
        }
    }

    results
}

/// Performs reconcile for a single directory. Ties everything else together.
async fn reconcile_dir(options: &Options, work_dir: &String, cache: &mut WorkspaceCache) -> Result<()> {
    println!("Processing path \"{}\".", work_dir);

    let (maybe_depot, maybe_workspace) = join!(
        run_p4_fstat_all(&options, &work_dir),
        gather_workspace(&options, &work_dir)
    );

    let depot: DepotState = maybe_depot?;
    let workspace: WorkspaceState = maybe_workspace?;

    if depot.file_records.len() == 0 && workspace.num_files == 0 {
        println!("The folder contains no files that need checking.");
        return Ok(());
    }

    println!("   Analyzing files for inconsistencies.");
    let start_time = Instant::now();

    //
    // Known cases. Only files added here will actually be changed in the end.
    // Every file must be added to either no or ONE of the below categories.
    //

    // Files in workspace, not in depot or deleted at have revision, but not checked out for add.
    let mut do_add = Vec::new();

    // Files in workspace, changed from have revision, but not checked out for edit.
    let mut do_edit = Vec::new();

    // Files in workspace, changed from have revision, but checked out for delete.
    let mut do_reopen_edit = Vec::new();

    // Files not in workspace, but not checked out for delete.
    let mut do_delete = Vec::new();

    // Files not in workspace, but checked out for edit.
    let mut do_reopen_delete = Vec::new();

    // Files not in workspace, but checked out for add.
    let mut do_revert_add = Vec::new();

    // Files in workspace, not changed from have revision, but checked out for edit.
    let mut do_revert_edit = Vec::new();

    // Files in workspace, not changed from have revision, but checked out for delete.
    let mut do_revert_delete = Vec::new();

    //
    // Cases that need digest computation to decide whether we should add them above.
    // We want to get as few as possible files here, but it's not always that nice.
    //

    // Files in workspace, maybe changed from have revision, but not checked out for edit.
    let mut check_edit = Vec::new();

    // Files in workspace, maybe not changed from have revision, but checked out for edit.
    let mut check_revert_edit = Vec::new();

    // Files in workspace, maybe not changed from have revision, but checked out for delete.
    let mut check_revert_delete_or_reopen_edit = Vec::new();

    //
    // Analysis phase one: Check depot records against workspace files.
    //

    use FileAction::*;

    for record in &depot.file_records {
        match record.head_action {
            // These are either irrelevant or will get caught in phase two, skip those.
            Some(Delete | MoveDelete) => (),
            // These exist in the depot, check if we still have them.
            Some(Add | Edit | MoveAdd | Branch | Integrate | Import | Purge) => {
                // The depot state may contain new files we haven't synced yet, skip those.
                if record.have_rev.is_some() {
                    match record.action {
                        // If we have them open for delete, but still have the file, revert that.
                        Some(Delete | MoveDelete) => {
                            if let Some(file) = workspace.get_filtered(&record.client_file_lower) {
                                if let (Some(digest_type), Some(size)) =
                                    (record.head_type, record.file_size)
                                {
                                    if size == file.size || digest_type != DigestType::Binary {
                                        check_revert_delete_or_reopen_edit
                                            .push((file, digest_type));

                                        if options.verbose {
                                            println!(
                                                "         File \"{}\" needs digest check for revert delete or reopen edit",
                                                &file.path
                                            );
                                        }
                                    } else {
                                        do_reopen_edit.push(record.client_file.clone());

                                        if options.verbose {
                                            println!(
                                                "         File \"{}\" has different length for reopen edit",
                                                &file.path
                                            );
                                        }
                                    }
                                } else {
                                    bail!(
                                        "Cannot handle \"{}\" 1",
                                        record.client_file
                                    );
                                }
                            }
                        }
                        // If we have them open for edit in some way, but don't have the file, reopen as delete.
                        Some(Edit | Integrate) => {
                            // No filter, adding a new ignore rule should not cause edits to reopen as deletions.
                            if !workspace.has_file(&record.client_file_lower) {
                                do_reopen_delete.push(record.client_file.clone());
                            }
                        }
                        // Otherwise, if we don't have the file, open as delete.
                        None => {
                            // No filter, adding a new ignore rule should not cause deletions.
                            if !workspace.has_file(&record.client_file_lower) {
                                do_delete.push(record.client_file.clone());
                            }
                        }
                        _ => bail!("Cannot handle \"{}\" 2", record.client_file),
                    }
                }
            }
            // These don't exist in the depot and can only be here because we opened them for add.
            None => match record.action {
                Some(Add | MoveAdd | Branch) => {
                    if !workspace.has_filtered(&record.client_file_lower) {
                        do_revert_add.push(record.client_file.clone());
                    }
                }
                _ => bail!("Cannot handle \"{}\" 3", record.client_file),
            },
            _ => bail!("Cannot handle \"{}\" 4", record.client_file),
        }
    }

    //
    // Analysis phase two: Check workspace files against depot records.
    //

    for file in workspace.files.iter().filter(|f| !f.filtered) {
        // First check if the file is present in the depot state.
        if let Some(record) = depot.get_client_record(&file.path_lower) {
            // Check what the depot states the file should be.
            match record.head_action {
                // We already took care of files we opened for add in phase one.
                None => (),
                // It's deleted at head, but we have the file.
                Some(Delete | MoveDelete) => {
                    match record.action {
                        // We already have it marked for add, skip.
                        Some(Add | MoveAdd | Branch) => (),
                        // We don't have it marked for add yet.
                        None => {
                            do_add.push(file.path.clone());
                        }
                        _ => bail!("Cannot handle \"{}\" 5", file.path),
                    }
                }
                // It already exists in the depot, check if we need to do something.
                Some(Add | Edit | MoveAdd | Branch | Integrate | Import | Purge) => {
                    match record.action {
                        // We should leave these alone as they can submit even with no changes made.
                        Some(Integrate) => (),
                        // We already took care of these in phase one.
                        Some(Delete | MoveDelete) => (),
                        // We have it open for edit, check if we reverted the change.
                        Some(Edit) => {
                            if let (Some(digest_type), Some(size)) =
                                (record.head_type, record.file_size)
                            {
                                if size == file.size || digest_type != DigestType::Binary {
                                    check_revert_edit.push((file, digest_type));

                                    if options.verbose {
                                        println!(
                                            "         File \"{}\" needs digest check for revert edit",
                                            &file.path
                                        );
                                    }
                                }
                            } else {
                                bail!("Cannot handle \"{}\" 6", file.path);
                            }
                        }
                        // We don't have it open, check if we should.
                        None => {
                            if let (Some(digest_type), Some(size)) =
                                (record.head_type, record.file_size)
                            {
                                if size != file.size && digest_type == DigestType::Binary {
                                    do_edit.push(file.path.clone());

                                    if options.verbose {
                                        println!(
                                            "         File \"{}\" has different length for edit",
                                            &file.path
                                        );
                                    }
                                } else {
                                    check_edit.push((file, digest_type));

                                    if options.verbose {
                                        println!(
                                            "         File \"{}\" needs digest check for edit",
                                            &file.path
                                        );
                                    }
                                }
                            }
                        }
                        _ => bail!("Cannot handle \"{}\" 7", file.path),
                    }
                }
                _ => bail!("Cannot handle \"{}\" 8", file.path),
            }
        } else {
            // The file is not in the depot at all and not ignored, mark for add.
            do_add.push(file.path.clone());
        }
    }

    println!(
        "      Analysis complete in {} seconds.",
        start_time.elapsed().as_secs_f32()
    );

    // Compute digests to see if we need to open files for edit.
    if check_edit.len() > 0 {
        println!("   Checking digests for {} files.", check_edit.len());

        let start_time = Instant::now();
        let mut total_size = 0;

        let results = parallel_compute_digests(check_edit, cache);
        for result in results {
            if !result.2 {
                total_size = total_size + result.0.size;
            }

            let record = depot.get_client_record(&result.0.path_lower).unwrap();
            if result.1 != record.digest.unwrap() {
                do_edit.push(result.0.path.clone());

                if options.verbose {
                    println!("         File \"{}\" digest is wrong.", &result.0.path);
                }
            }
        }

        if total_size > 0 {
            println!(
                "      Hashed {} in {} seconds.",
                format_size(total_size, BINARY),
                start_time.elapsed().as_secs_f32()
            );
        }
    }

    // Compute digests to see if we need to revert files open for edit.
    if check_revert_edit.len() > 0 {
        println!(
            "   Checking digests for {} files.",
            check_revert_edit.len()
        );

        let start_time = Instant::now();
        let mut total_size = 0;

        let results = parallel_compute_digests(check_revert_edit, cache);
        for result in results {
            if !result.2 {
                total_size = total_size + result.0.size;
            }

            let record = depot.get_client_record(&result.0.path_lower).unwrap();
            if result.1 == record.digest.unwrap() {
                do_revert_edit.push(result.0.path.clone());

                if options.verbose {
                    println!("         File \"{}\" digest is correct.", &result.0.path);
                }
            }
        }

        if total_size > 0 {
            println!(
                "      Hashed {} in {} seconds.",
                format_size(total_size, BINARY),
                start_time.elapsed().as_secs_f32()
            );
        }
    }

    // Compute digests to see if we need to revert deletes or reopen files for edit.
    if check_revert_delete_or_reopen_edit.len() > 0 {
        println!(
            "   Checking digests for {} files.",
            check_revert_delete_or_reopen_edit.len()
        );

        let start_time = Instant::now();
        let mut total_size = 0;

        let results = parallel_compute_digests(check_revert_delete_or_reopen_edit, cache);
        for result in results {
            if !result.2 {
                total_size = total_size + result.0.size;
            }

            let record = depot.get_client_record(&result.0.path_lower).unwrap();
            if result.1 == record.digest.unwrap() {
                do_revert_delete.push(result.0.path.clone());

                if options.verbose {
                    println!("         File \"{}\" digest is correct.", &result.0.path);
                }
            } else {
                do_reopen_edit.push(result.0.path.clone());

                if options.verbose {
                    println!("         File \"{}\" digest is wrong.", &result.0.path);
                }
            }
        }

        if total_size > 0 {
            println!(
                "      Hashed {} in {} seconds.",
                format_size(total_size, BINARY),
                start_time.elapsed().as_secs_f32()
            );
        }
    }

    let sum_changes = 0
        + do_add.len()
        + do_edit.len()
        + do_reopen_edit.len()
        + do_delete.len()
        + do_reopen_delete.len()
        + do_revert_add.len()
        + do_revert_edit.len()
        + do_revert_delete.len();

    if sum_changes <= 0 {
        println!("No changes to apply, everything up to date.");
        return Ok(());
    }

    if options.apply {
        println!("   Applying changes to p4.");
    } else {
        println!("   Counting changes (dry run).")
    }

    let start_time = Instant::now();

    // Add
    if do_add.len() > 0 {
        println!(
                "      Adding {} files in workspace, not in depot or deleted at have revision, but not checked out for add.",
                do_add.len()
            );

        if options.list {
            for file in &do_add {
                println!("         Add \"{}\".", &file);
            }
        }

        if options.apply {
            let args = ["add"];
            run_p4_command_batched(&options, &work_dir, &args, &do_add, true).await?;
        }
    }

    // Edit
    if do_edit.len() > 0 {
        println!(
                "      Editing {} files in workspace, changed from have revision, but not checked out for edit.",
                do_edit.len()
            );

        if options.list {
            for file in &do_edit {
                println!("         Edit \"{}\".", &file);
            }
        }

        if options.apply {
            let args = ["edit"];
            run_p4_command_batched(&options, &work_dir, &args, &do_edit, true).await?;
        }
    }

    // Reopen edit
    if do_reopen_edit.len() > 0 {
        println!(
                "      Revert+Editing {} files in workspace, changed from have revision, but checked out for delete.",
                do_reopen_edit.len()
            );

        if options.list {
            for file in &do_reopen_edit {
                println!("         Reopen Edit \"{}\".", &file);
            }
        }

        if options.apply {
            let args = ["revert", "-k"];
            run_p4_command_batched(&options, &work_dir, &args, &do_reopen_edit, false).await?;
            let args = ["edit"];
            run_p4_command_batched(&options, &work_dir, &args, &do_reopen_edit, true).await?;
        }
    }

    // Delete
    if do_delete.len() > 0 {
        println!(
            "      Deleting {} files not in workspace, but not checked out for delete.",
            do_delete.len()
        );

        if options.list {
            for file in &do_delete {
                println!("         Delete \"{}\".", &file);
            }
        }

        if options.apply {
            let args = ["delete", "-k"];
            run_p4_command_batched(&options, &work_dir, &args, &do_delete, true).await?;
        }
    }

    // Reopen delete
    if do_reopen_delete.len() > 0 {
        println!(
            "      Revert+Deleting {} files not in workspace, but checked out for edit.",
            do_reopen_delete.len()
        );

        if options.list {
            for file in &do_reopen_delete {
                println!("         Reopen Delete \"{}\".", &file);
            }
        }

        if options.apply {
            let args = ["revert", "-k"];
            run_p4_command_batched(&options, &work_dir, &args, &do_reopen_delete, false).await?;
            let args = ["delete", "-k"];
            run_p4_command_batched(&options, &work_dir, &args, &do_reopen_delete, true).await?;
        }
    }

    // Revert add
    if do_revert_add.len() > 0 {
        println!(
            "      Reverting {} files not in workspace, but checked out for add.",
            do_revert_add.len()
        );

        if options.list {
            for file in &do_revert_add {
                println!("         Revert Add \"{}\".", &file);
            }
        }

        if options.apply {
            let args = ["revert", "-k"];
            run_p4_command_batched(&options, &work_dir, &args, &do_revert_add, false).await?;
        }
    }

    // Revert edit
    if do_revert_edit.len() > 0 {
        println!(
                "      Reverting {} files in workspace, not changed from have revision, but checked out for edit.",
                do_revert_edit.len()
            );

        if options.list {
            for file in &do_revert_edit {
                println!("         Revert Edit \"{}\".", &file);
            }
        }

        if options.apply {
            let args = ["revert", "-k"];
            run_p4_command_batched(&options, &work_dir, &args, &do_revert_edit, false).await?;
        }
    }

    // Revert delete
    if do_revert_delete.len() > 0 {
        println!(
                "      Reverting {} files in workspace, not changed from have revision, but checked out for delete.",
                do_revert_delete.len()
            );

        if options.list {
            for file in &do_revert_delete {
                println!("         Revert Delete \"{}\".", &file);
            }
        }

        if options.apply {
            let args = ["revert", "-k"];
            run_p4_command_batched(&options, &work_dir, &args, &do_revert_delete, false).await?;
        }
    }

    if options.apply {
        println!(
            "      Applied {} changes in {} seconds.",
            sum_changes,
            start_time.elapsed().as_secs_f32()
        );
        println!("Inconsistencies fixed.");
    } else {
        println!(
            "      Counted {} changes in {} seconds.",
            sum_changes,
            start_time.elapsed().as_secs_f32()
        );
        println!("Inconsistencies found. Re-run with -a to apply changes.");
    }

    return Ok(());
}

fn main() -> Result<()> {
    let start_time = Instant::now();
    let mut options: Options = Options::parse();

    // Workspace input
    if options.workspace.is_none() {
        println!("No workspace passed, trying P4CLIENT.");
        options.workspace = env::var("P4CLIENT").ok();
    }

    match &options.workspace {
        None => bail!("No workspace found, use -w or set P4CLIENT."),
        Some(name) => println!("Using workspace \"{}\".", name),
    }

    // Changelist input
    match options.changelist {
        0 => println!("Using default pending changelist."),
        n => println!("Using pending changelist {}.", n),
    }

    // Verbose input
    if options.verbose {
        options.list = true;
    }

    let mut cache : WorkspaceCache = Default::default();

    // Load digest cache
    if let Some(proj_dirs) = ProjectDirs::from("com", "",  "FastReconcile") {
        let config = bincode::config::standard();
        let cache_path = proj_dirs.cache_dir().join("digests_".to_owned() + &options.workspace.clone().unwrap() + ".bin");

        if cache_path.exists() {
            println!("Loading cache from {}.", cache_path.display());
            let mut cache_file = File::open(cache_path)?;

            let mut buffer = Vec::new();
            cache_file.read_to_end(&mut buffer)?;

            let (decoded, _) : (WorkspaceCache, usize) = bincode::decode_from_slice(&buffer[..], config)?;

            cache = decoded;
            cache.out_of_date = false;

            println!("    Loaded {} cached digests.", cache.file_map.len());
        }
    }

    // Process paths from input, do these serially so the output makes more sense
    for original_path in &options.paths {
        // Correct the path since P4V tends to give us a bad one
        let mut path = original_path.trim_end_matches("\\...").to_string();
        if let Some(first_letter) = path.get_mut(0..1) {
            first_letter.make_ascii_uppercase();
        }
        // Use the path
        let check_path = PathBuf::from(&path);
        if check_path.exists() {
            if check_path.is_dir() {
                task::block_on(reconcile_dir(&options, &path, &mut cache))?;
            } else {
                println!("Skipping path \"{}\", is not a directory.", &path);
            }
        } else {
            println!("Skipping path \"{}\", doesn't exist.", &path);
        }
    }

    // Save digest cache
    if cache.out_of_date {
        if let Some(proj_dirs) = ProjectDirs::from("com", "",  "FastReconcile") {
            let config = bincode::config::standard();
            let cache_path = proj_dirs.cache_dir().join("digests_".to_owned() + &options.workspace.clone().unwrap() + ".bin");

            println!("Saving cache to {}.", cache_path.display());

            let prefix = cache_path.parent().unwrap();
            create_dir_all(prefix)?;

            let cache_file = File::create(cache_path)?;
            let mut cache_writer = BufWriter::new(cache_file);
            bincode::encode_into_std_write(&cache, &mut cache_writer, config)?;
            println!("    Saved {} cached digests.", cache.file_map.len());
        }
    }

    // We are done!
    println!(
        "Operation completed in {} seconds.",
        start_time.elapsed().as_secs_f32()
    );
    Ok(())
}
