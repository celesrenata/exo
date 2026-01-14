{ lib
, stdenv
, nodejs
, npmHooks
, fetchNpmDeps
, python3
}:

let
  dashboardServer = ./dashboard-server.js;
in

stdenv.mkDerivation rec {
  pname = "exo-dashboard";
  version = "0.3.0";

  src = lib.cleanSource ../../dashboard;

  npmDeps = fetchNpmDeps {
    src = lib.cleanSource ../../dashboard;
    hash = "sha256-koqsTfxfqJjo3Yq7x61q3duJ9Xtor/yOZcTjfBadZUs=";
  };

  nativeBuildInputs = [
    nodejs
    npmHooks.npmConfigHook
    python3
  ];

  configurePhase = ''
    runHook preConfigure
    
    # Copy source files to build directory
    cp -r $src/* .
    chmod -R +w .
    
    # Configure npm cache and install dependencies
    export HOME=$TMPDIR
    export npm_config_cache=$TMPDIR/npm-cache
    export npm_config_offline=true
    
    # Link npm dependencies
    ln -sf ${npmDeps}/node_modules ./node_modules
    
    runHook postConfigure
  '';

  buildPhase = ''
    runHook preBuild
    
    # Set NODE_ENV for production build
    export NODE_ENV=production
    
    # Run SvelteKit build process
    npm run build
    
    runHook postBuild
  '';

  installPhase = ''
        runHook preInstall
    
        # Install built dashboard static assets
        mkdir -p $out/share/exo/dashboard
    
        # Copy the built static files
        if [ -d "build" ]; then
          cp -r build/* $out/share/exo/dashboard/
        else
          echo "Error: build directory not found"
          exit 1
        fi
    
        # Create bin directory and install server script
        mkdir -p $out/bin
    
        # Copy the server script
        cp ${dashboardServer} $out/bin/exo-dashboard-server
        chmod +x $out/bin/exo-dashboard-server
    
        # Create a simple launcher script
        cat > $out/bin/exo-dashboard << 'EOF'
    #!/bin/sh
    DASHBOARD_DIR="OUTPUT_DIR_PLACEHOLDER/share/exo/dashboard"
    PORT=''${EXO_DASHBOARD_PORT:-8080}

    if [ ! -d "$DASHBOARD_DIR" ]; then
        echo "Error: Dashboard directory not found at $DASHBOARD_DIR"
        exit 1
    fi

    echo "Starting EXO Dashboard on port $PORT"
    echo "Dashboard files: $DASHBOARD_DIR"

    exec NODE_PLACEHOLDER OUTPUT_DIR_PLACEHOLDER/bin/exo-dashboard-server "$DASHBOARD_DIR"
    EOF

        # Fix the placeholders in the launcher script
        sed -i "s|OUTPUT_DIR_PLACEHOLDER|$out|g" $out/bin/exo-dashboard
        sed -i "s|NODE_PLACEHOLDER|${nodejs}/bin/node|g" $out/bin/exo-dashboard
        chmod +x $out/bin/exo-dashboard
    
        # Verify build output
        if [ ! -f "$out/share/exo/dashboard/index.html" ]; then
          echo "Warning: index.html not found in build output"
          ls -la $out/share/exo/dashboard/ || true
        fi
    
        runHook postInstall
  '';

  # Don't strip the JavaScript files
  dontStrip = true;

  meta = with lib; {
    description = "Web dashboard for EXO distributed AI inference";
    longDescription = ''
      A SvelteKit-based web dashboard for monitoring and managing EXO 
      distributed AI inference clusters. Provides real-time cluster topology 
      visualization, model management, and API access.
    '';
    homepage = "https://github.com/exo-explore/exo";
    license = licenses.asl20;
    maintainers = [ ];
    platforms = platforms.all;
  };
}
