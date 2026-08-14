package ocl;

import org.eclipse.emf.common.util.URI;
import org.eclipse.emf.ecore.EObject;
import org.eclipse.emf.ecore.EPackage;
import org.eclipse.emf.ecore.resource.Resource;
import org.eclipse.emf.ecore.resource.ResourceSet;
import org.eclipse.emf.ecore.resource.impl.ResourceSetImpl;
import org.eclipse.emf.ecore.xmi.impl.EcoreResourceFactoryImpl;
import org.eclipse.emf.ecore.xmi.impl.XMIResourceFactoryImpl;

import org.eclipse.ocl.pivot.ExpressionInOCL;
import org.eclipse.ocl.pivot.utilities.OCL;
import org.eclipse.ocl.pivot.utilities.OCLHelper;
import org.eclipse.ocl.pivot.utilities.ParserException;
import org.eclipse.ocl.xtext.completeocl.CompleteOCLStandaloneSetup;
import org.eclipse.ocl.xtext.oclinecore.OCLinEcoreStandaloneSetup;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

public class OCLValidator {

    public static void main(String[] args) throws Exception {

        // -- 1. Parse CLI arguments -----------------------------------------
        String ecorePath = null;
        String xmiPath   = null;
        String oclPath   = null;

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--ecore" -> ecorePath = args[++i];
                case "--xmi"   -> xmiPath   = args[++i];
                case "--ocl"   -> oclPath   = args[++i];
            }
        }

        if (ecorePath == null || xmiPath == null || oclPath == null) {
            emitFatalError("Missing arguments. Required: --ecore <path> --xmi <path> --ocl <path>");
            System.exit(1);
        }

        // -- Temporary diagnostics ------------------------------------------
        // System.err.println("DEBUG ecore        : " + ecorePath);
        // System.err.println("DEBUG xmi          : " + xmiPath);
        // System.err.println("DEBUG ocl          : " + oclPath);
        // System.err.println("DEBUG ecore exists : " + new File(ecorePath).exists());
        // System.err.println("DEBUG xmi   exists : " + new File(xmiPath).exists());
        // System.err.println("DEBUG ocl   exists : " + new File(oclPath).exists());
        // -------------------------------------------------------------------

        // -- 2. Bootstrap Eclipse OCL standalone ----------------------------
        CompleteOCLStandaloneSetup.doSetup();
        OCLinEcoreStandaloneSetup.doSetup();

        // -- 3. Load metamodel (.ecore) -------------------------------------
        ResourceSet rs = new ResourceSetImpl();
        rs.getResourceFactoryRegistry()
          .getExtensionToFactoryMap()
          .put("ecore", new EcoreResourceFactoryImpl());
        rs.getResourceFactoryRegistry()
          .getExtensionToFactoryMap()
          .put("xmi", new XMIResourceFactoryImpl());

        Resource ecoreResource = rs.getResource(
            URI.createFileURI(new File(ecorePath).getAbsolutePath()), true);
        ecoreResource.load(Collections.emptyMap());

        for (EObject obj : ecoreResource.getContents()) {
            if (obj instanceof EPackage pkg) {
                EPackage.Registry.INSTANCE.put(pkg.getNsURI(), pkg);
                registerSubPackages(pkg);
            }
        }

        // -- 4. Load instance model (.xmi) ----------------------------------
        Resource xmiResource = rs.getResource(
            URI.createFileURI(new File(xmiPath).getAbsolutePath()), true);
        xmiResource.load(Collections.emptyMap());

        // -- 5. Read OCL expressions ----------------------------------------
        List<String> expressions = Files.readAllLines(Path.of(oclPath))
            .stream()
            .map(String::trim)
            .filter(l -> !l.isEmpty() && !l.startsWith("--"))
            .toList();

        if (expressions.isEmpty()) {
            emitFatalError("No OCL expressions found in: " + oclPath);
            System.exit(1);
        }

        // -- 6. Validate ----------------------------------------------------
        OCL ocl = OCL.newInstance();
        List<Map<String, Object>> results = new ArrayList<>();

        for (EObject root : xmiResource.getContents()) {
            for (String expr : expressions) {
                results.add(evaluate(ocl, root, expr));
            }
        }

        // -- 7. Print JSON to stdout ----------------------------------------
        System.out.println(toJsonArray(results));
        ocl.dispose();

    } // end main

    // -------------------------------------------------------------------------
    // Evaluate one OCL invariant against one model object
    // -------------------------------------------------------------------------
    private static Map<String, Object> evaluate(OCL ocl, EObject root, String expr) {
        Map<String, Object> entry = new LinkedHashMap<>();
        String contextName = root.eClass().getEPackage().getName()
                           + "::" + root.eClass().getName();
        entry.put("constraint", expr);
        entry.put("context",    contextName);

        try {
            OCLHelper helper = ocl.createOCLHelper(root.eClass());
            ExpressionInOCL invariant = helper.createInvariant(expr);
            boolean ok = ocl.check(root, invariant);

            entry.put("satisfied", ok);
            if (ok) {
                entry.put("severity", "OK");
            } else {
                entry.put("severity", "ERROR");
                entry.put("message",
                    "Invariant violated on instance of " + contextName);
            }

        } catch (ParserException pe) {
            entry.put("satisfied", false);
            entry.put("severity",  "PARSE_ERROR");
            entry.put("message",   pe.getDiagnostic() != null
                                   ? pe.getDiagnostic().getMessage()
                                   : pe.getMessage());
        } catch (Exception e) {
            entry.put("satisfied", false);
            entry.put("severity",  "RUNTIME_ERROR");
            entry.put("message",   e.getClass().getSimpleName() + ": " + e.getMessage());
        }

        return entry;
    }

    // -------------------------------------------------------------------------
    // Recursively register nested EPackages
    // -------------------------------------------------------------------------
    private static void registerSubPackages(EPackage pkg) {
        for (EPackage sub : pkg.getESubpackages()) {
            EPackage.Registry.INSTANCE.put(sub.getNsURI(), sub);
            registerSubPackages(sub);
        }
    }

    // -------------------------------------------------------------------------
    // Minimal JSON serialiser - no external libraries required
    // -------------------------------------------------------------------------
    private static String toJsonArray(List<Map<String, Object>> list) {
        StringBuilder sb = new StringBuilder();
        sb.append("[\n");
        for (int i = 0; i < list.size(); i++) {
            sb.append(toJsonObject(list.get(i), "  "));
            if (i < list.size() - 1) sb.append(",");
            sb.append("\n");
        }
        sb.append("]");
        return sb.toString();
    }

    private static String toJsonObject(Map<String, Object> map, String indent) {
        StringBuilder sb = new StringBuilder();
        sb.append(indent).append("{\n");
        List<String> keys = new ArrayList<>(map.keySet());
        for (int i = 0; i < keys.size(); i++) {
            String key = keys.get(i);
            Object val = map.get(key);
            sb.append(indent).append("  ")
              .append(jsonString(key))
              .append(": ")
              .append(toJsonValue(val));
            if (i < keys.size() - 1) sb.append(",");
            sb.append("\n");
        }
        sb.append(indent).append("}");
        return sb.toString();
    }

    private static String toJsonValue(Object val) {
        if (val == null)            return "null";
        if (val instanceof Boolean) return val.toString();
        if (val instanceof Number)  return val.toString();
        return jsonString(val.toString());
    }

    private static String jsonString(String s) {
        StringBuilder sb = new StringBuilder("\"");
        for (char c : s.toCharArray()) {
            switch (c) {
                case '"'  -> sb.append("\\\"");
                case '\\' -> sb.append("\\\\");
                case '\n' -> sb.append("\\n");
                case '\r' -> sb.append("\\r");
                case '\t' -> sb.append("\\t");
                default   -> {
                    if (c < 0x20) {
                        sb.append(String.format("\\u%04x", (int) c));
                    } else {
                        sb.append(c);
                    }
                }
            }
        }
        sb.append("\"");
        return sb.toString();
    }

    // -------------------------------------------------------------------------
    // Emit a JSON fatal error to stdout
    // -------------------------------------------------------------------------
    private static void emitFatalError(String msg) {
        System.out.println("{\"fatal_error\": " + jsonString(msg) + "}");
    }
}