{
  description = "EXO: Run your own AI cluster at home with everyday devices";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    # Provides Rust dev-env integration:
    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    # Provides formatting infrastructure:
    treefmt-nix = {
      url = "github:numtide/treefmt-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  # TODO: figure out caching story
  # nixConfig = {
  #   # nix community cachix
  #   extra-trusted-public-keys = "nix-community.cachix.org-1:mB9FSh9qf2dCimDSUo8Zy7bkq5CX+/rkCWyvRCYg3Fs=";
  #   extra-substituters = "https://nix-community.cachix.org";
  # };

  outputs =
    inputs:
    let
      systems = [
        "x86_64-linux"
        "aarch64-darwin"
        "aarch64-linux"
      ];
      fenixToolchain = system: inputs.fenix.packages.${system}.complete;
      
      # Package builder function
      mkExoPackage = { pkgs, system, accelerator ? "cpu" }: 
        let
          python = pkgs.python313;
          rustToolchain = fenixToolchain system;
        in
        pkgs.stdenv.mkDerivation rec {
          pname = "exo-${accelerator}";
          version = "0.1.0";
          
          src = ./.;
          
          nativeBuildInputs = with pkgs; [
            python
            rustToolchain.rustc
            rustToolchain.cargo
            nodejs
            pkg-config
          ] ++ pkgs.lib.optionals pkgs.stdenv.isLinux [
            openssl.dev
          ];
          
          buildInputs = with pkgs; [
            python
            openssl
          ] ++ pkgs.lib.optionals (accelerator == "cuda") [
            cudaPackages.cudatoolkit
            cudaPackages.cudnn
          ] ++ pkgs.lib.optionals (accelerator == "rocm") [
            rocmPackages.hip
            rocmPackages.rocm-runtime
          ];
          
          buildPhase = ''
            # Build dashboard
            cd dashboard
            npm ci
            npm run build
            cd ..
            
            # Build Rust bindings
            cd rust/exo_pyo3_bindings
            cargo build --release
            cd ../..
            
            # Install Python package
            ${python}/bin/python -m pip install --prefix=$out .
          '';
          
          installPhase = ''
            mkdir -p $out/bin $out/share/exo
            
            # Copy dashboard build
            cp -r dashboard/build $out/share/exo/dashboard
            
            # Copy Rust bindings
            cp rust/exo_pyo3_bindings/target/release/*.so $out/lib/python*/site-packages/ || true
            
            # Create wrapper script
            cat > $out/bin/exo << EOF
            #!${pkgs.bash}/bin/bash
            export PYTHONPATH="$out/lib/python*/site-packages:\$PYTHONPATH"
            export EXO_DASHBOARD_PATH="$out/share/exo/dashboard"
            exec ${python}/bin/python -m exo "\$@"
            EOF
            chmod +x $out/bin/exo
          '';
          
          meta = with pkgs.lib; {
            description = "Run your own AI cluster at home with everyday devices";
            homepage = "https://github.com/exo-explore/exo";
            license = licenses.asl20;
            maintainers = [ ];
            platforms = platforms.unix;
          };
        };
    in
    {
      # NixOS module
      nixosModules.default = { config, lib, pkgs, ... }:
        with lib;
        let
          cfg = config.services.exo;
        in
        {
          options.services.exo = {
            enable = mkEnableOption "EXO distributed AI inference system";
            
            package = mkOption {
              type = types.package;
              default = pkgs.exo-cpu or (mkExoPackage { inherit pkgs; system = pkgs.system; accelerator = "cpu"; });
              description = "EXO package to use";
            };
            
            accelerator = mkOption {
              type = types.enum [ "cpu" "cuda" "rocm" "intel" "mlx" ];
              default = "cpu";
              description = "Hardware accelerator to use";
            };
            
            port = mkOption {
              type = types.port;
              default = 52415;
              description = "Port for EXO dashboard and API";
            };
            
            openFirewall = mkOption {
              type = types.bool;
              default = false;
              description = "Open firewall for EXO communication";
            };
            
            user = mkOption {
              type = types.str;
              default = "exo";
              description = "User to run EXO services";
            };
            
            group = mkOption {
              type = types.str;
              default = "exo";
              description = "Group to run EXO services";
            };
          };
          
          config = mkIf cfg.enable {
            users.users.${cfg.user} = {
              isSystemUser = true;
              group = cfg.group;
              description = "EXO system user";
            };
            
            users.groups.${cfg.group} = {};
            
            systemd.services.exo = {
              description = "EXO distributed AI inference system";
              after = [ "network.target" ];
              wantedBy = [ "multi-user.target" ];
              
              serviceConfig = {
                Type = "simple";
                User = cfg.user;
                Group = cfg.group;
                ExecStart = "${cfg.package}/bin/exo";
                Restart = "always";
                RestartSec = "10";
                
                # Security settings
                NoNewPrivileges = true;
                PrivateTmp = true;
                ProtectSystem = "strict";
                ProtectHome = true;
                ReadWritePaths = [ "/var/lib/exo" ];
              };
              
              environment = {
                EXO_PORT = toString cfg.port;
              };
            };
            
            networking.firewall = mkIf cfg.openFirewall {
              allowedTCPPorts = [ cfg.port ];
            };
            
            systemd.tmpfiles.rules = [
              "d /var/lib/exo 0755 ${cfg.user} ${cfg.group} -"
            ];
          };
        };
        
      # Overlays
      overlays.default = final: prev: {
        exo-cpu = mkExoPackage { pkgs = final; system = final.system; accelerator = "cpu"; };
        exo-cuda = mkExoPackage { pkgs = final; system = final.system; accelerator = "cuda"; };
        exo-rocm = mkExoPackage { pkgs = final; system = final.system; accelerator = "rocm"; };
        exo-intel = mkExoPackage { pkgs = final; system = final.system; accelerator = "intel"; };
        exo-mlx = mkExoPackage { pkgs = final; system = final.system; accelerator = "mlx"; };
      };
    } //
    inputs.flake-utils.lib.eachSystem systems (
      system:
      let
        pkgs = import inputs.nixpkgs {
          inherit system;
          overlays = [ inputs.fenix.overlays.default ];
        };
        treefmtEval = inputs.treefmt-nix.lib.evalModule pkgs {
          projectRootFile = "flake.nix";
          programs.ruff-format.enable = true;
          programs.ruff-format.excludes = [ "rust/exo_pyo3_bindings/exo_pyo3_bindings.pyi" ];
          programs.rustfmt.enable = true;
          programs.rustfmt.package = (fenixToolchain system).rustfmt;
          programs.nixpkgs-fmt.enable = true;
        };
      in
      {
        # Packages
        packages = {
          default = mkExoPackage { inherit pkgs system; };
          exo-cpu = mkExoPackage { inherit pkgs system; accelerator = "cpu"; };
          exo-cuda = mkExoPackage { inherit pkgs system; accelerator = "cuda"; };
          exo-rocm = mkExoPackage { inherit pkgs system; accelerator = "rocm"; };
          exo-intel = mkExoPackage { inherit pkgs system; accelerator = "intel"; };
          exo-mlx = mkExoPackage { inherit pkgs system; accelerator = "mlx"; };
        };
        
        formatter = treefmtEval.config.build.wrapper;
        checks.formatting = treefmtEval.config.build.check inputs.self;
        checks.lint = pkgs.runCommand "lint-check" { } ''
          export RUFF_CACHE_DIR="$TMPDIR/ruff-cache"
          ${pkgs.ruff}/bin/ruff check ${inputs.self}/
          touch $out
        '';

        devShells.default = pkgs.mkShell {
          packages =
            with pkgs;
            [
              # PYTHON
              python313
              uv
              ruff
              basedpyright

              # RUST
              ((fenixToolchain system).withComponents [
                "cargo"
                "rustc"
                "clippy"
                "rustfmt"
                "rust-src"
              ])
              rustup # Just here to make RustRover happy

              # NIX
              nixpkgs-fmt

              # SVELTE
              nodejs

              # MISC
              just
              jq
            ]
            ++ (pkgs.lib.optionals pkgs.stdenv.isLinux [
              # IFCONFIG
              unixtools.ifconfig

              # Build dependencies for Linux
              pkg-config
              openssl
            ])
            ++ (pkgs.lib.optionals pkgs.stdenv.isDarwin [
              # MACMON
              macmon
            ]);

          shellHook = ''
            # PYTHON
            export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${pkgs.python313}/lib"
            ${pkgs.lib.optionalString pkgs.stdenv.isLinux ''
              # Build environment for Linux
              export PKG_CONFIG_PATH="${pkgs.openssl.dev}/lib/pkgconfig:$PKG_CONFIG_PATH"
              export LD_LIBRARY_PATH="${pkgs.openssl.out}/lib:$LD_LIBRARY_PATH"
            ''}
            echo
            echo "🍎🍎 Run 'just <recipe>' to get started"
            just --list
          '';

        };
      }
    );
}
