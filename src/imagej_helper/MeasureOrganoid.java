import ij.IJ;
import ij.ImagePlus;
import ij.gui.Roi;
import ij.gui.PolygonRoi;
import ij.io.RoiEncoder;
import ij.io.RoiDecoder;
import ij.measure.Calibration;
import ij.measure.Measurements;
import ij.measure.ResultsTable;
import ij.plugin.filter.Analyzer;
import ij.plugin.filter.ThresholdToSelection;
import ij.process.ImageProcessor;
import java.nio.file.*;

// Runs headlessly using the same ij.jar installed with Fiji. No Fiji GUI state is changed.
public class MeasureOrganoid {
    public static void main(String[] args) throws Exception {
        ImagePlus image=IJ.openImage(args[0]);
        if(image==null) throw new IllegalArgumentException("Cannot open mask");
        Roi roi;
        if(args.length>4 && args[4].endsWith(".csv")) {
            java.util.List<String> lines=Files.readAllLines(Path.of(args[4]));
            float[] xs=new float[lines.size()-1], ys=new float[lines.size()-1];
            for(int i=1;i<lines.size();i++) {
                String[] xy=lines.get(i).split(",");
                xs[i-1]=Float.parseFloat(xy[0]);ys[i-1]=Float.parseFloat(xy[1]);
            }
            roi=new PolygonRoi(xs,ys,xs.length,Roi.POLYGON);
        }
        else if(args.length>4) roi=RoiDecoder.open(args[4]);
        else {
            ImageProcessor ip=image.getProcessor();
            ip.setThreshold(255,255,ImageProcessor.NO_LUT_UPDATE);
            roi=new ThresholdToSelection().convert(ip);
            ip.resetThreshold();
        }
        if(roi==null) throw new IllegalArgumentException("No area ROI");
        Calibration cal=new Calibration();
        cal.pixelWidth=Double.parseDouble(args[2]);
        cal.pixelHeight=Double.parseDouble(args[3]);
        cal.setUnit("micron"); image.setCalibration(cal); image.setRoi(roi);
        ResultsTable table=new ResultsTable();
        int flags=Measurements.AREA|Measurements.PERIMETER|Measurements.SHAPE_DESCRIPTORS|Measurements.FERET;
        new Analyzer(image,flags,table).measure();
        String[] headings={"Area","Perim.","Circ.","AR","Round","Solidity","Feret","MinFeret"};
        String[] names={"Area","Perimeter","Circularity","AspectRatio","Roundness","Solidity","Feret","MinFeret"};
        StringBuilder json=new StringBuilder("{");
        for(int i=0;i<names.length;i++) {
            double value=table.getValue(headings[i],0);
            if(!Double.isFinite(value)) throw new IllegalArgumentException("Nonfinite measurement: "+names[i]);
            if(i>0) json.append(",");
            json.append("\"").append(names[i]).append("\":").append(value);
        }
        json.append("}");
        Path output=Path.of(args[1]); Files.createDirectories(output);
        Files.writeString(output.resolve("measurements.json"),json.toString());
        RoiEncoder.save(roi,output.resolve("predicted.roi").toString());
        if(args.length>4) {
            ImageProcessor reference=image.getProcessor().createProcessor(image.getWidth(),image.getHeight());
            reference.setValue(255);reference.setRoi(roi);reference.fill(roi.getMask());
            javax.imageio.ImageIO.write(reference.getBufferedImage(),"png",output.resolve("reference_mask.png").toFile());
        }
        image.close();
    }
}
